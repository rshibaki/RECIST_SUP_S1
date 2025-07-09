import numpy as np
import torch 

def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    dtype: torch.dtype = torch.float,
    dim: int = 1
): # -> torch.Tensor
    # if 'dim' is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - labels.dim())
        labels = torch.reshape(labels, shape)
        
    sh = list(labels.shape)
    
    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")
    
    sh[dim] = num_classes
    
    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    # logging.debug(f"Tensor 'o' size: {o.size()}, Tensor 'labels' size: {labels.size()}") ##
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)
    
    return labels 



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


def calc_dice(y_true, y_pred, epsilon=1e-8, dice_type="jaccard"):
    assert y_true.shape == y_pred.shape
    if dice_type == "jaccard":
        denom_true = np.sum(y_true * y_true)
        denom_pred = np.sum(y_pred * y_pred)
    else:
        denom_true = np.sum(y_true)
        denom_pred = np.sum(y_pred)

    denom = denom_true + denom_pred
    numer = 2 * np.sum(y_true * y_pred)

    dice_scores = numer / (denom + epsilon)

    return dice_scores


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
