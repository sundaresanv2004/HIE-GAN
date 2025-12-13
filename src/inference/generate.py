import argparse
import sys
import torch
import trimesh
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from models.feature_extractor import FeatureExtractor
from models.explicit_branch import ExplicitDeformer
from models.init_mesh import create_sphere_mesh
from utils.config import load_configs
from dataloader.dataset import DatasetLoader, ShapeNetDataset

def load_model(config_dir, checkpoint_path, device):
    _, model_cfg, _ = load_configs(config_dir)
    
    encoder = FeatureExtractor(out_dim=model_cfg["feature_extractor"]["out_dim"]).to(device)
    
    V0, E0 = create_sphere_mesh(model_cfg["explicit_branch"]["init_mesh"]["subdivisions"])
    V0, E0 = V0.to(device), E0.to(device)
    
    hidden_dims = model_cfg["explicit_branch"].get("gcn_hidden_dims", [128, 64])
    explicit = ExplicitDeformer(init_mesh=(V0, E0), feature_dim=model_cfg["feature_extractor"]["out_dim"], hidden_dims=hidden_dims).to(device)

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    explicit.load_state_dict(checkpoint["explicit_state_dict"])
    
    encoder.eval()
    explicit.eval()
    
    return encoder, explicit, model_cfg

def generate_single(image_path, output_dir, encoder, explicit, device, model_cfg):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        raw_image = Image.open(img_path).convert("RGB")
        img_tensor = transform(raw_image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return

    with torch.no_grad():
        feat = encoder(img_tensor)
        pred_verts = explicit(feat)
    
    pred_verts = pred_verts.squeeze(0).cpu().numpy()
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = img_path.stem + "_pred.obj"
    out_path = out_dir / out_name
    
    # Save Mesh
    temp_sphere = trimesh.creation.icosphere(subdivisions=model_cfg["explicit_branch"]["init_mesh"]["subdivisions"])
    mesh = trimesh.Trimesh(vertices=pred_verts, faces=temp_sphere.faces)
    mesh.export(out_path)
    
    # Save Input Image
    out_img_path = out_dir / (img_path.stem + "_input.png")
    raw_image.save(out_img_path)
    
    return out_path

def generate_batch(config_dir, output_dir, encoder, explicit, device, model_cfg, num_samples=5):
    # Load dataset configuration
    dataset_cfg, _, train_cfg = load_configs(config_dir)
    
    # Use full dataset
    class Args:
        data_root = None
        num_samples = None 
        val_split = 0.0
        seed = 42
        debug = False
        pin_memory = False
    
    args = Args()
    
    # Dummy logger
    class Logger:
        def info(self, m): pass
        def warning(self, m): pass
        
    loader = DatasetLoader(dataset_cfg, train_cfg, args, Logger())
    
    # Inspect dataset directly to access class info
    # We initialize the dataset manually to get easy access to paths and labels
    dataset = ShapeNetDataset(
        root_dir=dataset_cfg["root_dir"],
        classes=dataset_cfg["classes"],
        pc_filename=dataset_cfg["pointcloud"]["filename"],
        image_size=dataset_cfg["image"]["size"]
    )
    
    print(f"Found {len(dataset)} items. Generating {num_samples} samples per class...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by class
    from collections import defaultdict
    class_indices = defaultdict(list)
    
    # Iterate dataset indices to find class association
    for idx, obj_path_str in enumerate(dataset.object_paths):
        # obj_path is root/class_id/model_id
        # We want class_id
        path_obj = Path(obj_path_str)
        class_id = path_obj.parent.name
        
        # Check if we need more samples for this class
        if len(class_indices[class_id]) < num_samples:
            class_indices[class_id].append(idx)
            
    # Now generate
    temp_sphere = trimesh.creation.icosphere(subdivisions=model_cfg["explicit_branch"]["init_mesh"]["subdivisions"])
    
    total_generated = 0
    
    # Process by class
    for class_name, indices in class_indices.items():
        print(f"Processing class: {class_name} ({len(indices)} samples)")
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in tqdm(indices, desc=class_name, leave=False):
            # item is (image_tensor, gt_pc)
            img_tensor, _ = dataset[idx] 
            
            # Get model ID for filename
            obj_path = Path(dataset.object_paths[idx])
            model_id = obj_path.name
            
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = encoder(img_tensor)
                pred_verts = explicit(feat)
                
            vert = pred_verts.squeeze(0).cpu().numpy()
            
            # Save Mesh
            mesh = trimesh.Trimesh(vertices=vert, faces=temp_sphere.faces)
            mesh.export(class_dir / f"{model_id}.obj")
            
            # Save Input Image
            # Un-normalize: input = (input * std) + mean
            inv_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
            inv_std = [1/0.229, 1/0.224, 1/0.225]
            
            inv_normalize = transforms.Normalize(mean=inv_mean, std=inv_std)
            
            # Clone to avoid modifying tensor if used elsewhere (not here, but safety)
            img_vis = img_tensor.squeeze(0).clone().cpu()
            img_vis = inv_normalize(img_vis)
            img_vis = torch.clamp(img_vis, 0, 1)
            
            # Convert to PIL
            img_pil = transforms.ToPILImage()(img_vis)
            img_pil.save(class_dir / f"{model_id}_input.png")
            
            total_generated += 1

    print(f"✓ Generated {total_generated} models in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="HIE-GAN Generation Tool")
    parser.add_argument("--mode", type=str, choices=["single", "all"], required=True, help="Generation mode")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config-dir", type=str, default="src/configs", help="Config directory")
    parser.add_argument("--output-dir", type=str, default="output/generated", help="Output directory")
    parser.add_argument("--image", type=str, help="Input image for single mode")
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    encoder, explicit, model_cfg = load_model(args.config_dir, args.checkpoint, device)
    
    if args.mode == "single":
        if not args.image:
            print("❌ --image is required for single mode")
            return
        path = generate_single(args.image, args.output_dir, encoder, explicit, device, model_cfg)
        print(f"✓ Saved to {path}")
        
    elif args.mode == "all":
        # Generate for all classes (here we limit to a reasonable number per class to avoid flooding)
        # The user said "generate 3d model for all the class". We'll interpret this as generating samples into folders.
        generate_batch(args.config_dir, args.output_dir, encoder, explicit, device, model_cfg, num_samples=20) # Limit to 20 for safety

if __name__ == "__main__":
    main()
