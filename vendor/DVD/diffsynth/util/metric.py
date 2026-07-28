# Author: Bingxin Ke
# Last modified: 2024-02-15


import pandas as pd
import torch


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / \
            self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def pixel_mean(pred, gt, valid_mask):
    if valid_mask is not None:
        masked_pred = pred * valid_mask
        masked_gt = gt * valid_mask

        valid_pixel_count = torch.sum(valid_mask, dim=(0, 1))

        pred_mean = torch.sum(masked_pred, dim=(0, 1)) / valid_pixel_count
        gt_mean = torch.sum(masked_gt, dim=(0, 1)) / valid_pixel_count
    else:
        pred_mean = torch.mean(pred, dim=(0, 1))
        gt_mean = torch.mean(gt, dim=(0, 1))

    mean_difference = torch.abs(pred_mean - gt_mean)
    return mean_difference


def pixel_var(pred, gt, valid_mask):
    if valid_mask is not None:
        masked_pred = pred * valid_mask
        masked_gt = gt * valid_mask

        valid_pixel_count = torch.sum(valid_mask, dim=(0, 1))

        pred_mean = torch.sum(masked_pred, dim=(0, 1)) / valid_pixel_count
        gt_mean = torch.sum(masked_gt, dim=(0, 1)) / valid_pixel_count

        pred_var = torch.sum(valid_mask * (pred - pred_mean)
                             ** 2, dim=(0, 1)) / valid_pixel_count
        gt_var = torch.sum(valid_mask * (gt - gt_mean)**2,
                           dim=(0, 1)) / valid_pixel_count
    else:
        pred_var = torch.var(pred, dim=(0, 1))
        gt_var = torch.var(gt, dim=(0, 1))

    var_difference = torch.abs(pred_var - gt_var)

    return var_difference


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(
        actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    # print(f"total mask: {n}")
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    # print(f"abs_relative_diff: {abs_relative_diff}")
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss



def relative_temporal_diff(pred, gt, valid_mask=None, eps=1e-6):
    """
    pred, gt: [F, H, W]
    valid_mask: [F, H, W] (bool)
    """

    # relative temporal difference
    pred_rel = (pred[1:] - pred[:-1]) / (pred[:-1] + eps)  # [F-1, H, W]
    gt_rel = (gt[1:] - gt[:-1]) / (gt[:-1] + eps)

    diff = pred_rel - gt_rel

    if valid_mask is not None:
        # AND 两帧 mask
        valid_pair = valid_mask[1:] & valid_mask[:-1]
        diff[~valid_pair] = 0
        n = valid_pair.sum((-1, -2))  # [F-1]
    else:
        n = diff.shape[-1] * diff.shape[-2]

    diff2 = diff ** 2
    # diff1 = torch.abs(diff)
    # l1 = torch.sum(diff1, (-1, -2)) / (n + eps)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()
    # return rmse.mean()


def boundary_metrics(pred_depth, rgb, valid_mask=None,
                             th_depth_ratio=1.05, th_rgb_grad=0.15,
                             tolerance=1, eps=1e-6):
    import torch
    import torch.nn.functional as F

    device = pred_depth.device

    pred_depth, valid_mask = pred_depth.unsqueeze(1), valid_mask.unsqueeze(1)

    if rgb.shape[1] == 3:
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    else:
        gray = rgb

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=device, dtype=rgb.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=device, dtype=rgb.dtype).view(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + eps)

    B = mag.shape[0]
    mag_flat = mag.view(B, -1)
    mag_min = mag_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    mag_max = mag_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    mag_norm = (mag - mag_min) / (mag_max - mag_min + eps)

    edges_gt = (mag_norm > th_rgb_grad).float()

    d = pred_depth.clamp(min=eps)

    def get_edge_with_nms(ratio_map, dim):
        is_candidate = ratio_map > th_depth_ratio

        if dim == 3:
            k_size, pad = (1, 3), (0, 1)
        else:
            k_size, pad = (3, 1), (1, 0)

        local_max = F.max_pool2d(
            ratio_map, kernel_size=k_size, stride=1, padding=pad)
        is_peak = (ratio_map == local_max)

        return is_candidate & is_peak

    d_pad = F.pad(d, (1, 1, 1, 1), mode='replicate')  # [B, 1, H+2, W+2]
    d_center = d

    # Right: d(x+1, y) / d(x, y)
    ratio_right = d_pad[:, :, 1:-1, 2:] / d_center
    mask_right = get_edge_with_nms(ratio_right, dim=3)

    # Left: d(x-1, y) / d(x, y)
    ratio_left = d_pad[:, :, 1:-1, :-2] / d_center
    mask_left = get_edge_with_nms(ratio_left, dim=3)

    # Bottom: d(x, y+1) / d(x, y)
    ratio_bottom = d_pad[:, :, 2:, 1:-1] / d_center
    mask_bottom = get_edge_with_nms(ratio_bottom, dim=2)

    # Top: d(x, y-1) / d(x, y)
    ratio_top = d_pad[:, :, :-2, 1:-1] / d_center
    mask_top = get_edge_with_nms(ratio_top, dim=2)

    edges_pred = (mask_right | mask_left | mask_bottom | mask_top).float()

    if valid_mask is not None:
        edges_gt = edges_gt * valid_mask
        edges_pred = edges_pred * valid_mask

    if tolerance > 0:
        kernel_size = 2 * tolerance + 1
        edges_gt_dilated = F.max_pool2d(
            edges_gt, kernel_size=kernel_size, stride=1, padding=tolerance)
        edges_pred_dilated = F.max_pool2d(
            edges_pred, kernel_size=kernel_size, stride=1, padding=tolerance)
    else:
        edges_gt_dilated = edges_gt
        edges_pred_dilated = edges_pred

    # True Positives
    tp_prec = (edges_pred * edges_gt_dilated).sum()
    tp_rec = (edges_gt * edges_pred_dilated).sum()

    # Totals
    n_pred = edges_pred.sum()
    n_gt = edges_gt.sum()

    precision = tp_prec / (n_pred + eps)
    recall = tp_rec / (n_gt + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)

    return {
        "f1": f1_score.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        # "edges_pred": edges_pred,
        # "edges_gt": edges_gt
    }
