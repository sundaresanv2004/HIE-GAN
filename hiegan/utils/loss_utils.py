# hiegan/utils/loss_utils.py
import torch
from pytorch3d.loss import chamfer_distance


def chamfer_loss(pred_points, gt_points):
    """
    Compute Chamfer Distance between predicted and ground-truth point clouds.

    pred_points: (B, N, 3)
    gt_points: (B, M, 3)
    """
    loss, _ = chamfer_distance(pred_points, gt_points)
    return loss


def occupancy_iou(pred_sdf, sample_xyz, threshold=0.0):
    """
    Compute IoU of occupancy grids from predicted SDF values.

    pred_sdf: (B, N, 1) predicted SDF at sampled points
    sample_xyz: (B, N, 3) corresponding 3D points
    threshold: SDF threshold to consider occupied
    """
    pred_occ = (pred_sdf <= threshold).float()  # 1 if inside mesh
    # For simplicity, assume ground-truth occupancy = 1 for all points in point cloud
    gt_occ = torch.ones_like(pred_occ)
    intersection = (pred_occ * gt_occ).sum(dim=1)
    union = ((pred_occ + gt_occ) > 0).sum(dim=1)
    iou = intersection / (union + 1e-8)
    return iou.mean()


def f1_score(pred_points, gt_points, threshold=0.01):
    """
    Compute F1 score between predicted and GT point clouds.
    threshold: distance tolerance to count as correct
    """
    B = pred_points.shape[0]
    f1_list = []
    for b in range(B):
        pd = pred_points[b]  # (N,3)
        gt = gt_points[b]  # (M,3)
        # pairwise distances
        dist_matrix = torch.cdist(pd, gt)  # (N,M)
        # precision
        precision = (dist_matrix.min(dim=1)[0] < threshold).float().mean()
        # recall
        recall = (dist_matrix.min(dim=0)[0] < threshold).float().mean()
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_list.append(f1)
    return torch.stack(f1_list).mean()
