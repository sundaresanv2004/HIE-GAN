import torch
from torch.utils.data import DataLoader
from dataloader.dataset import ShapeNetDataset
from models.feature_extractor import FeatureExtractor
from models.explicit_branch import ExplicitDeformer
from models.init_mesh import create_sphere_mesh
from losses.chamfer import chamfer_loss


def train_phase1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- dataset --------
    dataset = ShapeNetDataset(
        root_dir="/workspace/data",
        transform=None  # you will add transforms later
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # -------- model --------
    encoder = FeatureExtractor(out_dim=256).to(device)

    # initial sphere
    V0, E0 = create_sphere_mesh(num_subdivisions=2)
    V0 = V0.to(device)
    E0 = E0.to(device)

    explicit = ExplicitDeformer(
        init_mesh=(V0, E0),
        feature_dim=256
    ).to(device)

    optim = torch.optim.Adam(
        list(encoder.parameters()) + list(explicit.parameters()),
        lr=1e-4
    )

    for epoch in range(20):
        for img, gt_pc in loader:
            img = img.to(device)  # (B, C, H, W)
            gt_pc = gt_pc.to(device)  # (B, Ng, 3)

            # forward:
            feat = encoder(img)  # (B,256)
            pred_pc = explicit(feat)  # (B,Nv,3)

            loss = chamfer_loss(pred_pc, gt_pc)

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"epoch={epoch} loss={loss.item():.4f}")


if __name__ == "__main__":
    train_phase1()
