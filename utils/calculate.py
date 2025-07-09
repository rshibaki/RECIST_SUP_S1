from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial.distance import directed_hausdorff, cdist

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Literal, Sequence
from monai.metrics.utils import (
    get_edge_surface_distance,
    ignore_background,
    prepare_spacing,
)
from monai.utils import (
    convert_data_type,
)
from monai.metrics.hausdorff_distance import _compute_percentile_hausdorff_distance

#Dice score関数
def calc_dice(y_true, y_pred, epsilon=1e-8):
    assert y_true.shape == y_pred.shape

    denom_true = np.sum(y_true)
    denom_pred = np.sum(y_pred)

    denom = denom_true + denom_pred
    if denom_true == 0:
        dice_score = 'undefined'
    
    else:
        numer = 2 * np.sum(y_true * y_pred)

        dice_score = numer / denom

    return dice_score #Dice score関数


# 表面だけを残す処理
def surface_voxels(mask):
    return np.logical_xor(mask, binary_erosion(mask))


#ハウスドルフ距離の関数
def hausdorff_metrics(pred, gt, spacing):
    pred_surf = surface_voxels(pred)
    gt_surf = surface_voxels(gt)

    pred_coords = np.argwhere(pred_surf) * spacing
    gt_coords = np.argwhere(gt_surf) * spacing

    # Directed HDs
    dists_pred_to_gt = np.min(np.linalg.norm(pred_coords[:, None] - gt_coords[None, :], axis=2), axis=1)
    dists_gt_to_pred = np.min(np.linalg.norm(gt_coords[:, None] - pred_coords[None, :], axis=2), axis=1)

    all_dists = np.concatenate([dists_pred_to_gt, dists_gt_to_pred])
    
    hd_max = np.max(all_dists)
    hd_95 = np.percentile(all_dists, 95)
    hd_mean = np.mean(all_dists)
    assd = (np.mean(dists_pred_to_gt) + np.mean(dists_gt_to_pred)) / 2

    return hd_max, hd_mean, hd_95, assd



def compute_hausdorff_distance(
    y_pred: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: float | None = None,
    directed: bool = False,
    spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[int | float]] | None = None,
    mode: Literal["max", "percentile", "mean"] = "max",  # ★追加
) -> torch.Tensor:
    """
    This implementation https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/metrics/hausdorff_distance.py#L138-L199
    Compute the Hausdorff distance between prediction and ground truth.
    Selectable distance type: max (standard HD), percentile (HD95, etc.), or mean (MSD).
    """
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)
    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    batch_size, n_class = y_pred.shape[:2]
    hd = torch.empty((batch_size, n_class), dtype=torch.float, device=y_pred.device)

    img_dim = y_pred.ndim - 2
    spacing_list = prepare_spacing(spacing=spacing, batch_size=batch_size, img_dim=img_dim)

    for b, c in np.ndindex(batch_size, n_class):
        _, distances, _ = get_edge_surface_distance(
            y_pred[b, c],
            y[b, c],
            distance_metric=distance_metric,
            spacing=spacing_list[b],
            symmetric=not directed,
            class_index=c,
        )

        if mode == "percentile":
            if percentile is None:
                raise ValueError("Percentile must be specified when mode='percentile'")
            value = torch.max(torch.stack([
                _compute_percentile_hausdorff_distance(d, percentile) for d in distances
            ]))
        elif mode == "mean":
            value = torch.mean(torch.stack([
                torch.mean(d) for d in distances if d.numel() > 0
            ]))
        else:  # "max"
            value = torch.max(torch.stack([
                torch.max(d) for d in distances if d.numel() > 0
            ]))

        hd[b, c] = value

    return hd


#############################################################################################

def extract_bb(msk, to_open):
    assert msk.dtype == np.bool_, msk.dtype
    assert msk.ndim in [3, 4], msk.shape
    if msk.ndim == 4:
        assert msk.shape[-1] == 1, msk.shape
    z, y, x = msk.shape[:3]
    z = np.any(msk, axis=(1, 2))
    z_min, z_max = np.nonzero(z)[0][[0, -1]]
    y = np.any(msk[z_min : z_max + 1], axis=(0, 2))
    y_min, y_max = np.nonzero(y)[0][[0, -1]]
    x = np.any(msk[z_min : z_max + 1, y_min : y_max + 1], axis=(0, 1))
    x_min, x_max = np.nonzero(x)[0][[0, -1]]

    if to_open:
        z_max += 1
        y_max += 1
        x_max += 1
    return np.array([z_min, y_min, x_min, z_max, y_max, x_max])


def merge_bbox(bb_zyxzyx1, bb_zyxzyx2):
    zmin = min(bb_zyxzyx1[0], bb_zyxzyx2[0])
    ymin = min(bb_zyxzyx1[1], bb_zyxzyx2[1])
    xmin = min(bb_zyxzyx1[2], bb_zyxzyx2[2])
    zmax = max(bb_zyxzyx1[3], bb_zyxzyx2[3])
    ymax = max(bb_zyxzyx1[4], bb_zyxzyx2[4])
    xmax = max(bb_zyxzyx1[5], bb_zyxzyx2[5])
    return np.array([zmin, ymin, xmin, zmax, ymax, xmax])


def crop_input(input, bb_zyxzyx):
    input = input[
        bb_zyxzyx[0] : bb_zyxzyx[3],
        bb_zyxzyx[1] : bb_zyxzyx[4],
        bb_zyxzyx[2] : bb_zyxzyx[5],
    ]
    return input


def check_intersect(bb_zyxzyx1, bb_zyxzyx2):
    for dim in range(3):
        if not max(bb_zyxzyx1[dim], bb_zyxzyx2[dim]) <= min(
            bb_zyxzyx1[dim + 3], bb_zyxzyx2[dim + 3]
        ):
            break
    else:
        return True
    return False

def get_3d_line_coordinates(p1_zyx, p2_zyx):
    """
    (x-x1)/dx = (y-y1)/dy = (z-z1)/dz
    """

    p1_zyx = np.array(p1_zyx)
    p2_zyx = np.array(p2_zyx)

    z1, y1, x1 = p1_zyx
    dz, dy, dx = p2_zyx - p1_zyx

    idx = np.argmax(np.abs([dz, dy, dx]))

    min_val, max_val = sorted([p1_zyx[idx], p2_zyx[idx]])
    return_point = False
    if idx == 0:  # z
        if dz == 0:
            return_point = True
        else:
            z_range = np.arange(min_val, max_val + 1)
            x_range = (z_range - z1) / dz * dx + x1
            y_range = (z_range - z1) / dz * dy + y1
    elif idx == 1:  # y
        if dy == 0:
            return_point = True
        else:
            y_range = np.arange(min_val, max_val + 1)
            x_range = (y_range - y1) / dy * dx + x1
            z_range = (y_range - y1) / dy * dz + z1
    elif idx == 2:  # x
        if dx == 0:
            return_point = True
        else:
            x_range = np.arange(min_val, max_val + 1)
            y_range = (x_range - x1) / dx * dy + y1
            z_range = (x_range - x1) / dx * dz + z1
    if return_point:
        z_range = np.array(z1)
        y_range = np.array(y1)
        x_range = np.array(x1)
    zyx = np.stack([z_range, y_range, x_range]).T
    return zyx  # (b, 3)