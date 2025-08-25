import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from tqdm import tqdm
from functools import partial
import torch.multiprocessing as mp

# --- Import your project modules ---
from hiegan.config import Config
from hiegan.dataset import ShapeNetMVDataset
from hiegan.utils.collate import mesh_collate_fn
from hiegan.utils.mesh_utils import mesh_to_pointcloud
from hiegan.utils.loss_utils import HieGanLoss
from hiegan.models.generator import HIEGenerator
from hiegan.models.discriminator import PointCloudDiscriminator  # Your new discriminator name

# ---------------------- Config & Device ----------------------
cfg = Config()
device = torch.device(cfg.DEVICE)
print("Using device:", device)
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

# ---------------------- Dataset & Loader (Corrected) ----------------------
# This uses the efficient collate function we defined before.
ds = ShapeNetMVDataset(cfg.DATASET_ROOT, image_size=cfg.IMAGE_SIZE)
collate_with_device = partial(mesh_collate_fn, device=device)
dl = DataLoader(
    ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
    num_workers=cfg.NUM_WORKERS, collate_fn=collate_with_device,
    pin_memory=True, drop_last=True
)
print(f"Dataset size: {len(ds)}")

# ---------------------- Models & Template Mesh (Corrected) ----------------------
G = HIEGenerator(latent_dim=cfg.LATENT_DIM).to(device)
D = PointCloudDiscriminator(num_points=cfg.NUM_POINTS_SAMPLED).to(device)

# CORRECTED: Use a fixed icosphere as the template mesh
template_mesh = ico_sphere(level=3, device=device)
template_verts = template_mesh.verts_list()[0]
template_faces = template_mesh.faces_list()[0]
template_edges = template_mesh.edges_packed().t()  # Get correct edge index

# ---------------------- Optimizers & Loss (Corrected) ----------------------
G_optimizer = Adam(G.parameters(), lr=cfg.LR_G)
D_optimizer = Adam(D.parameters(), lr=cfg.LR_D)
criterion = HieGanLoss(cfg)

# ---------------------- Training Loop (Corrected) ----------------------
for epoch in range(cfg.EPOCHS):
    G.train()
    D.train()

    pbar = tqdm(dl, desc=f"Epoch {epoch + 1}/{cfg.EPOCHS}")
    for imgs, gt_meshes in pbar:
        # Data is already on the correct device thanks to collate_fn
        # imgs shape is already (B, C, H, W)

        # --- 1. Train Discriminator ---
        D_optimizer.zero_grad()

        # Get real points from ground truth meshes
        real_points = mesh_to_pointcloud(gt_meshes, cfg.NUM_POINTS_SAMPLED)

        # Generate fake points (no gradients needed for generator here)
        with torch.no_grad():
            gen_outputs = G(imgs, template_verts, template_edges)
            deformed_verts = gen_outputs["deformed_vertices"]
            fake_meshes = Meshes(verts=deformed_verts, faces=template_faces.expand(cfg.BATCH_SIZE, -1, -1))
            fake_points = mesh_to_pointcloud(fake_meshes, cfg.NUM_POINTS_SAMPLED)

        # Get discriminator logits
        disc_real_logits = D(real_points)
        disc_fake_logits = D(fake_points.detach())  # Detach to be safe

        # Compute loss and update discriminator
        d_losses = criterion.get_discriminator_loss(disc_real_logits, disc_fake_logits)
        d_losses["d_loss"].backward()
        D_optimizer.step()

        # --- 2. Train Generator ---
        G_optimizer.zero_grad()

        # Generate fake meshes again, this time tracking gradients for the generator
        gen_outputs = G(imgs, template_verts, template_edges)
        deformed_verts = gen_outputs["deformed_vertices"]
        fake_meshes_for_g = Meshes(verts=deformed_verts, faces=template_faces.expand(cfg.BATCH_SIZE, -1, -1))
        fake_points_for_g = mesh_to_pointcloud(fake_meshes_for_g, cfg.NUM_POINTS_SAMPLED)

        # Get discriminator's (updated) opinion on the fake meshes
        disc_fake_logits_for_g = D(fake_points_for_g)

        # Compute loss and update generator
        g_losses = criterion.get_generator_loss(gen_outputs, gt_meshes, disc_fake_logits_for_g)
        g_losses["g_loss"].backward()
        G_optimizer.step()

        # --- Update Progress Bar ---
        pbar.set_postfix({
            "G_Loss": f"{g_losses['g_loss'].item():.4f}",
            "D_Loss": f"{d_losses['d_loss'].item():.4f}",
            "Chamfer": f"{g_losses['chamfer'].item():.4f}"
        })

    # --- Save Checkpoint ---
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"hiegan_epoch_{epoch + 1}.pth")
        torch.save({
            'generator_state_dict': G.state_dict(),
            'discriminator_state_dict': D.state_dict(),
        }, checkpoint_path)
        print(f"\nCheckpoint saved at {checkpoint_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    train()
