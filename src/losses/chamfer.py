import torch


def chamfer_loss(pred_pc, gt_pc):
    """
    pred_pc: (B, Nv, 3)
    gt_pc:   (B, Ng, 3)
    """
    # pairwise distances
    dist = torch.cdist(pred_pc, gt_pc)  # (B, Nv, Ng)

    # nearest neighbor from pred to gt
    d1 = dist.min(dim=2)[0]  # (B, Nv)

    # nearest neighbor from gt to pred
    d2 = dist.min(dim=1)[0]  # (B, Ng)

    return d1.mean() + d2.mean()
