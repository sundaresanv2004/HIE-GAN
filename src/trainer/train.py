import os
import torch
from torch.utils.data import DataLoader

from utils.config import load_configs
from utils.logger import setup_logger, CSVLogger

from dataloader.dataset import ShapeNetDataset
from models.feature_extractor import FeatureExtractor
from models.explicit_branch import ExplicitDeformer
from models.init_mesh import create_sphere_mesh
from losses.chamfer import chamfer_loss


def train_phase1():
    dataset_cfg, model_cfg, train_cfg = load_configs()

    logger = setup_logger(
        train_cfg["logging"]["log_dir"],
        train_cfg["logging"]["log_filename"]
    )
    csvlogger = CSVLogger(
        train_cfg["logging"]["log_dir"],
        train_cfg["logging"]["csv_filename"]
    )

    device = torch.device(
        model_cfg["device"] if torch.cuda.is_available() else "cpu"
    )

    dataset = ShapeNetDataset(
        root_dir=dataset_cfg["root_dir"],
        classes=dataset_cfg["classes"],
        pc_filename=dataset_cfg["pointcloud"]["filename"],
        image_size=dataset_cfg["image"]["size"],
    )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True
    )

    encoder = FeatureExtractor(out_dim=model_cfg["feature_extractor"]["out_dim"]).to(device)

    V0, E0 = create_sphere_mesh(
        model_cfg["explicit_branch"]["init_mesh"]["subdivisions"]
    )
    V0, E0 = V0.to(device), E0.to(device)

    explicit = ExplicitDeformer(
        init_mesh=(V0, E0),
        feature_dim=model_cfg["feature_extractor"]["out_dim"],
        hidden_dims=model_cfg["explicit_branch"]["gcn_hidden_dims"],
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(explicit.parameters()),
        lr=train_cfg["optimizer"]["lr"]
    )

    ckpt_dir = train_cfg["checkpoints"]["dir"]
    ckpt_file = train_cfg["checkpoints"]["filename"]
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)

    start_epoch = 0

    if train_cfg["checkpoints"]["resume"] and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"])
        explicit.load_state_dict(ckpt["explicit"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from checkpoint: {ckpt_path}")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        for step, (img, gt_pc) in enumerate(loader):
            img = img.to(device)
            gt_pc = gt_pc.to(device)
            feat = encoder(img)
            pred_pc = explicit(feat)
            loss = chamfer_loss(pred_pc, gt_pc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % train_cfg["logging"]["log_every_n_steps"] == 0:
                logger.info(f"Epoch {epoch}, Step {step}, Loss {loss.item():.6f}")
                csvlogger.write(epoch, step, loss.item())

        torch.save({
            "encoder": encoder.state_dict(),
            "explicit": explicit.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }, ckpt_path)

        logger.info(f"Checkpoint saved at {ckpt_path}")

    logger.info("Training Finished.")
