import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import modules
from hiegan.config import Config
from hiegan.dataset import ShapeNetMVDataset
from hiegan.utils.mesh_utils import load_and_normalize_mesh, mesh_to_pointcloud
from hiegan.utils.render_utils import render_mesh_rgb, simple_renderer
from hiegan.models.generator import HIEGenerator
from hiegan.models.discriminator import PointCloudDiscriminator
from hiegan.utils.loss_utils import chamfer_loss, occupancy_iou, f1_score

# Optional: Chamfer Distance from PyTorch3D
from pytorch3d.loss import chamfer_distance

# ---------------------- Config & Device ----------------------
cfg = Config()
device = torch.device(cfg.DEVICE if torch.cuda.is_available() or cfg.DEVICE == "mps" else "cpu")
print("Using device:", device)

# ---------------------- Dataset & Loader ----------------------
dataset_path = "../data/ShapenetCoreV2"
ds = ShapeNetMVDataset(dataset_path, image_size=cfg.IMAGE_SIZE, multi_view=cfg.MULTI_VIEW)
dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

print(f"Dataset size: {len(ds)}")

# ---------------------- Models ----------------------
G = HIEGenerator(latent_dim=cfg.LATENT_DIM, device=str(device)).to(device)
D = PointCloudDiscriminator().to(device)

# ---------------------- Optimizers ----------------------
G_optimizer = optim.Adam(G.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
D_optimizer = optim.Adam(D.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

# ---------------------- Loss Functions ----------------------
bce_loss = nn.BCELoss()

# ---------------------- Checkpoint directory ----------------------
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

# ---------------------- Training Loop ----------------------
for epoch in range(cfg.EPOCHS):
    G.train()
    D.train()
    epoch_recon_loss = 0.0
    epoch_adv_loss = 0.0
    epoch_chamfer = 0.0
    epoch_iou = 0.0
    epoch_f1 = 0.0

    pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
    for imgs, mesh_paths in pbar:
        imgs = imgs[:,0].to(device)
        batch_size = imgs.shape[0]

        # Load template mesh and edges (for explicit branch)
        template_mesh = load_and_normalize_mesh(mesh_paths[0], device=str(device))
        template_vertices = template_mesh.verts_list()[0]
        edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long).to(device)  # replace with actual mesh edges

        # Sample random points for implicit SDF
        sample_xyz = torch.rand(batch_size, 1024, 3, device=device)*2 -1

        # ------------------ Generator forward ------------------
        G_out = G(imgs, template_mesh_vertices=template_vertices, template_mesh_edges=edge_index, sample_xyz=sample_xyz)
        implicit_sdf = G_out['implicit_sdf']
        fused_vertices = G_out['fused_vertices']

        # ------------------ Compute reconstruction losses ------------------
        gt_mesh = load_and_normalize_mesh(mesh_paths[0], device=str(device))
        gt_points = mesh_to_pointcloud(gt_mesh, num_samples=1024)
        pred_points = fused_vertices.unsqueeze(1) if fused_vertices is not None else sample_xyz

        recon_loss = chamfer_loss(pred_points, gt_points)
        iou = occupancy_iou(implicit_sdf, sample_xyz)
        f1 = f1_score(pred_points, gt_points)

        # ------------------ Discriminator update ------------------
        D_optimizer.zero_grad()
        D_real = D(gt_points)
        D_fake = D(pred_points.detach())
        real_labels = torch.ones(batch_size,1,device=device)
        fake_labels = torch.zeros(batch_size,1,device=device)
        D_loss = nn.BCELoss()(D_real, real_labels) + nn.BCELoss()(D_fake, fake_labels)
        D_loss.backward()
        D_optimizer.step()

        # ------------------ Generator update ------------------
        G_optimizer.zero_grad()
        D_fake_for_G = D(pred_points)
        adv_loss = nn.BCELoss()(D_fake_for_G, real_labels)
        G_total_loss = recon_loss + 0.1*adv_loss
        G_total_loss.backward()
        G_optimizer.step()

        # ------------------ Accumulate metrics ------------------
        epoch_recon_loss += recon_loss.item()
        epoch_adv_loss += adv_loss.item()
        epoch_chamfer += recon_loss.item()
        epoch_iou += iou.item()
        epoch_f1 += f1.item()

        pbar.set_postfix({
            "Recon": f"{recon_loss.item():.4f}",
            "Adv": f"{adv_loss.item():.4f}",
            "IoU": f"{iou.item():.4f}",
            "F1": f"{f1.item():.4f}"
        })

    # ------------------ Save checkpoint ------------------
    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"hiegan_epoch{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'optimizer_G_state_dict': G_optimizer.state_dict(),
        'optimizer_D_state_dict': D_optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    # ------------------ Visualization ------------------
    G.eval()
    with torch.no_grad():
        fused_mesh_viz = G_out['fused_vertices'][0].cpu()
        renderer = simple_renderer(device=str(device))
        # Render as image
        mesh_img = render_mesh_rgb(template_mesh, renderer, device=str(device))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4,4))
        plt.imshow(mesh_img[0].cpu().numpy())
        plt.axis("off")
        plt.title(f"Epoch {epoch+1} Preview\nChamfer: {epoch_chamfer/len(dl):.4f}, IoU: {epoch_iou/len(dl):.4f}, F1: {epoch_f1/len(dl):.4f}")
        plt.show()

