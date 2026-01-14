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
from models.implicit_branch import ImplicitDecoder
from models.fusion_module import FusionModule
from models.init_mesh import create_sphere_mesh
from utils.config import load_configs
from utils.mesh_ops import generate_mesh_from_sdf
from dataloader.dataset import DatasetLoader, ShapeNetDataset

def load_model(config_dir, checkpoint_path, device):
    _, model_cfg, _ = load_configs(config_dir)
    
    # Feature Extractor
    encoder = FeatureExtractor(out_dim=model_cfg["feature_extractor"]["out_dim"]).to(device)
    
    # Explicit Branch
    V0, E0 = create_sphere_mesh(model_cfg["explicit_branch"]["init_mesh"]["subdivisions"])
    V0, E0 = V0.to(device), E0.to(device)
    
    hidden_dims = model_cfg["explicit_branch"].get("gcn_hidden_dims", [128, 64])
    explicit = ExplicitDeformer(
        init_mesh=(V0, E0), 
        feature_dim=model_cfg["feature_extractor"]["out_dim"], 
        hidden_dims=hidden_dims,
        use_layer_norm=model_cfg["explicit_branch"].get("use_layer_norm", True)
    ).to(device)

    # Implicit Branch (Phase 2)
    imp_cfg = model_cfg["implicit_branch"]
    implicit = ImplicitDecoder(
        feature_dim=model_cfg["feature_extractor"]["out_dim"],
        hidden_dim=imp_cfg["hidden_dim"],
        num_layers=imp_cfg["num_layers"],
        skip_connection_at=imp_cfg.get("skip_connection_at", []),
        use_positional_encoding=imp_cfg.get("use_positional_encoding", True),
        pos_enc_levels=imp_cfg.get("pos_enc_levels", 6)
    ).to(device)

    # Fusion Module (Phase 3)
    fusion = FusionModule(step_size=1.0).to(device)

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    explicit.load_state_dict(checkpoint["explicit_state_dict"])
    
    if "implicit_state_dict" in checkpoint:
        implicit.load_state_dict(checkpoint["implicit_state_dict"])
    
    if "fusion_state_dict" in checkpoint:
        fusion.load_state_dict(checkpoint["fusion_state_dict"])
    else:
        print("⚠ Warning: No fusion state found (Phase 1/2 checkpoint?)")
    
    encoder.eval()
    explicit.eval()
    implicit.eval()
    fusion.eval()
    
    return encoder, explicit, implicit, fusion, model_cfg

def generate_single(image_path, output_dir, encoder, explicit, implicit, fusion, device, model_cfg):
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

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = img_path.stem
    
    with torch.no_grad():
        feat = encoder(img_tensor) # (1, C)
        
        # 1. Explicit Mesh
        pred_verts_exp = explicit(feat)
        
        # 2. Fused Mesh (Phase 3)
        # Refines explicit verts using implicit gradients
        # Must enable grad for SDF normal computation
        with torch.enable_grad():
             pred_verts_fused = fusion(pred_verts_exp, implicit, feat)
        
        # Export Explicit
        vert_exp = pred_verts_exp.squeeze(0).cpu().numpy()
        temp_sphere = trimesh.creation.icosphere(subdivisions=model_cfg["explicit_branch"]["init_mesh"]["subdivisions"])
        mesh_explicit = trimesh.Trimesh(vertices=vert_exp, faces=temp_sphere.faces)
        mesh_explicit.export(out_dir / f"{out_name}_explicit.obj")

        # Export Fused
        vert_fused = pred_verts_fused.squeeze(0).cpu().numpy()
        mesh_fused = trimesh.Trimesh(vertices=vert_fused, faces=temp_sphere.faces)
        mesh_fused.export(out_dir / f"{out_name}_fused.obj")
        
        # 3. Implicit Mesh (SDF)
        mesh_implicit = generate_mesh_from_sdf(implicit, feat, resolution=64, threshold=0.0, device=device)
        if mesh_implicit:
            mesh_implicit.export(out_dir / f"{out_name}_implicit.obj")
            
    # Save Input Image
    raw_image.save(out_dir / (img_path.stem + "_input.png"))
    
    return out_dir

def generate_batch(config_dir, output_dir, encoder, explicit, implicit, fusion, device, model_cfg, num_samples=5):
    dataset_cfg, _, train_cfg = load_configs(config_dir)
    
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

    from collections import defaultdict
    class_indices = defaultdict(list)
    
    for idx, obj_path_str in enumerate(dataset.object_paths):
        path_obj = Path(obj_path_str)
        class_id = path_obj.parent.name
        if len(class_indices[class_id]) < num_samples:
            class_indices[class_id].append(idx)
            
    temp_sphere = trimesh.creation.icosphere(subdivisions=model_cfg["explicit_branch"]["init_mesh"]["subdivisions"])
    total_generated = 0
    
    for class_name, indices in class_indices.items():
        print(f"Processing class: {class_name} ({len(indices)} samples)")
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in tqdm(indices, desc=class_name, leave=False):
            img_tensor, _, _, _ = dataset[idx]
            
            obj_path = Path(dataset.object_paths[idx])
            model_id = obj_path.name
            
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = encoder(img_tensor)
                
                # Explicit
                pred_verts_exp = explicit(feat)
                vert_exp = pred_verts_exp.squeeze(0).cpu().numpy()
                mesh_exp = trimesh.Trimesh(vertices=vert_exp, faces=temp_sphere.faces)
                mesh_exp.export(class_dir / f"{model_id}_explicit.obj")
                
                # Fused (Phase 3)
                pred_verts_fused = fusion(pred_verts_exp, implicit, feat)
                vert_fused = pred_verts_fused.squeeze(0).cpu().numpy()
                mesh_fused = trimesh.Trimesh(vertices=vert_fused, faces=temp_sphere.faces)
                mesh_fused.export(class_dir / f"{model_id}_fused.obj")

                # Implicit
                mesh_imp = generate_mesh_from_sdf(implicit, feat, resolution=64, device=device)
                if mesh_imp:
                     mesh_imp.export(class_dir / f"{model_id}_implicit.obj")
            
            # Save Input
            inv_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
            inv_std = [1/0.229, 1/0.224, 1/0.225]
            inv_normalize = transforms.Normalize(mean=inv_mean, std=inv_std)
            
            img_vis = img_tensor.squeeze(0).clone().cpu()
            img_vis = inv_normalize(img_vis)
            img_vis = torch.clamp(img_vis, 0, 1)
            transforms.ToPILImage()(img_vis).save(class_dir / f"{model_id}_input.png")
            
            total_generated += 1

    print(f"✓ Generated {total_generated} models in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="HIE-GAN Generation Tool (Phase 3)")
    parser.add_argument("--mode", type=str, choices=["single", "all"], required=True, help="Generation mode")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config-dir", type=str, default="src/configs", help="Config directory")
    parser.add_argument("--output-dir", type=str, default="output/generated", help="Output directory")
    parser.add_argument("--image", type=str, help="Input image for single mode")
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    encoder, explicit, implicit, fusion, model_cfg = load_model(args.config_dir, args.checkpoint, device)
    
    if args.mode == "single":
        if not args.image:
            print("❌ --image is required for single mode")
            return
        generate_single(args.image, args.output_dir, encoder, explicit, implicit, fusion, device, model_cfg)
        
    elif args.mode == "all":
        generate_batch(args.config_dir, args.output_dir, encoder, explicit, implicit, fusion, device, model_cfg, num_samples=20)

if __name__ == "__main__":
    main()
