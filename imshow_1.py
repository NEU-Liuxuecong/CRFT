import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import json
import flow_viz


# ---------------- Utility Functions ---------------- #
def _resolve_checkpoint(path: str) -> str:
    """If the specified path exists, return it directly; otherwise, fallback to the latest .pth file in the checkpoints directory."""
    if osp.exists(path):
        return path
    ckpt_dir = osp.join(osp.dirname(__file__), 'checkpoints')
    if not osp.isdir(ckpt_dir):
        raise FileNotFoundError(f'Checkpoint not found: {path} and directory missing {ckpt_dir}')
    pths = [osp.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not pths:
        raise FileNotFoundError('No .pth files found in checkpoints directory')
    pths.sort(key=lambda p: osp.getmtime(p), reverse=True)
    print('Fallback to using the latest weights:', osp.basename(pths[0]))
    return pths[0]


def PoinsTransf(pts: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply affine matrix h (2x3) to point set pts[[x,y],...]."""
    pts_h = np.concatenate([pts.astype(np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)  # [N,3]
    out = (h.astype(np.float32) @ pts_h.T).T  # [N,2]
    return out


def PoinsTransf_flow(pts: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Displace corner points using pixel-level optical flow (2,H,W) by directly sampling the corresponding pixel displacement."""
    if flow.shape[0] != 2:
        raise ValueError('flow must be of shape [2,H,W]')
    H, W = flow.shape[1], flow.shape[2]
    pts_int = pts.astype(int)
    pts_int[:, 0] = np.clip(pts_int[:, 0], 0, W - 1)
    pts_int[:, 1] = np.clip(pts_int[:, 1], 0, H - 1)
    disp = flow[:, pts_int[:, 1], pts_int[:, 0]].T  # [N,2]
    return pts.astype(np.float32) + disp.astype(np.float32)


def estimate_affine_from_flow(flow: torch.Tensor, step: int = 8) -> np.ndarray:
    """Estimate least squares affine matrix H (2x3) from dense optical flow."""
    if not isinstance(flow, torch.Tensor):
        raise TypeError(f"flow must be of type torch.Tensor, but got {type(flow)}")
    
    if flow.dim() != 3 or flow.shape[0] != 2:
        raise ValueError(f"flow has incorrect dimensions, expected [2, H, W], but got {flow.shape}")

    flow_np = flow.detach().cpu().numpy()
    H, W = flow_np.shape[1], flow_np.shape[2]

    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid spatial dimensions for flow, H and W must be greater than 0, but got H={H}, W={W}")

    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    xs_sample = xs[::step, ::step].reshape(-1)
    ys_sample = ys[::step, ::step].reshape(-1)
    u_sample = flow_np[0, ::step, ::step].reshape(-1)
    v_sample = flow_np[1, ::step, ::step].reshape(-1)

    # Ensure sufficient sample size
    if xs_sample.size < 6:
        raise ValueError(f"Insufficient valid sample points, at least 6 points are required, but got {xs_sample.size}")

    ones = np.ones_like(xs_sample)
    A = np.stack([xs_sample, ys_sample, ones], axis=1)  # [N,3]
    bx = xs_sample + u_sample
    by = ys_sample + v_sample

    # Least squares solution for affine parameters
    px, *_ = np.linalg.lstsq(A, bx, rcond=None)
    py, *_ = np.linalg.lstsq(A, by, rcond=None)

    H_mat = np.stack([px, py], axis=0).astype(np.float32)  # [2,3]

    if H_mat.shape != (2, 3):
        raise ValueError(f"Calculated affine matrix H has incorrect format, expected [2, 3], but got {H_mat.shape}")

    return H_mat


def show_pic_with_polygon(tensor_img: torch.Tensor, poly_ref: np.ndarray, poly_other: np.ndarray, subplot_idx: int, total: int):
    """Display grayscale image + overlay two polygons. tensor_img: [1,1,H,W] or [1,C,H,W]."""
    if tensor_img.dim() == 4:
        # Take the first channel for display
        img = tensor_img[0, 0].cpu().numpy()
    elif tensor_img.dim() == 3:
        img = tensor_img[0].cpu().numpy()
    else:
        img = tensor_img.cpu().numpy()
    img_u8 = img.astype(np.float32)
    if img_u8.max() <= 1.0:
        img_u8 = (img_u8 * 255.0).round()
    img_u8 = img_u8.astype(np.uint8)
    rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    # Swap colors: poly_ref to yellow, poly_other to red
    if poly_ref is not None:
        cv2.polylines(rgb, [poly_ref.astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
    if poly_other is not None:
        cv2.polylines(rgb, [poly_other.astype(np.int32)], True, (255, 0, 0), 2, cv2.LINE_AA)
    ax = plt.subplot(1, total, subplot_idx)
    ax.imshow(rgb)
    ax.axis('off')


def show_flow_arrow(image_t: torch.Tensor, flow_t: torch.Tensor, step: int = 32):
    """Display optical flow with sparse arrows. image_t: [1,C,H,W] flow_t: [2,H,W]"""
    if image_t.dim() == 4:
        img = image_t[0, 0].cpu().numpy()
    else:
        img = image_t[0].cpu().numpy()
    if img.max() <= 1.0:
        img_disp = (img * 255.0).round().astype(np.uint8)
    else:
        img_disp = img.astype(np.uint8)
    rgb = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)
    flow_np = flow_t.detach().cpu().numpy()
    H, W = flow_np.shape[1], flow_np.shape[2]
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    X, Y = np.meshgrid(xs, ys)
    U = flow_np[0, ys[:, None], xs[None, :]]
    V = flow_np[1, ys[:, None], xs[None, :]]
    plt.imshow(rgb)
    plt.quiver(X, Y, U, V, color='r', angles='xy', scale_units='xy', scale=1.0, width=0.003)
    plt.axis('off')


def PoinsTransf(pts: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply affine matrix h (2x3) to point array pts[[x,y], ...]."""
    pts = np.asarray(pts, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    out = (h @ pts_h.T).T
    return out


def estimate_affine_from_flow(flow: torch.Tensor, step: int = 8) -> np.ndarray:
    """Estimate affine matrix from dense flow via least squares."""
    if not isinstance(flow, torch.Tensor):
        raise TypeError(f"flow must be of type torch.Tensor, but got {type(flow)}")
    
    if flow.dim() != 3 or flow.shape[0] != 2:
        raise ValueError(f"flow has incorrect dimensions, expected [2, H, W], but got {flow.shape}")

    flow_np = flow.detach().cpu().numpy()
    H, W = flow_np.shape[1], flow_np.shape[2]

    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid spatial dimensions for flow, H and W must be greater than 0, but got H={H}, W={W}")

    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    xs_sample = xs[::step, ::step].reshape(-1)
    ys_sample = ys[::step, ::step].reshape(-1)
    u_sample = flow_np[0, ::step, ::step].reshape(-1)
    v_sample = flow_np[1, ::step, ::step].reshape(-1)

    # Ensure sufficient sample size
    if xs_sample.size < 6:
        raise ValueError(f"Insufficient valid sample points, at least 6 points are required, but got {xs_sample.size}")

    ones = np.ones_like(xs_sample)
    A = np.stack([xs_sample, ys_sample, ones], axis=1)
    bx = xs_sample + u_sample
    by = ys_sample + v_sample

    px, *_ = np.linalg.lstsq(A, bx, rcond=None)
    py, *_ = np.linalg.lstsq(A, by, rcond=None)
    return np.stack([px, py], axis=0).astype(np.float32)

# New: Sparse arrow optical flow
def show_flow_arrow(image_t: torch.Tensor, flow_t: torch.Tensor, step: int = 32):
    """Display optical flow with sparse arrows. image_t: [1,C,H,W] flow_t: [2,H,W]"""
    if image_t.dim() == 4:
        img = image_t[0, 0].cpu().numpy()
    else:
        img = image_t[0].cpu().numpy()
    if img.max() <= 1.0:
        img_disp = (img * 255.0).round().astype(np.uint8)
    else:
        img_disp = img.astype(np.uint8)
    rgb = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)
    flow_np = flow_t.detach().cpu().numpy()
    H, W = flow_np.shape[1], flow_np.shape[2]
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    X, Y = np.meshgrid(xs, ys)
    U = flow_np[0, ys[:, None], xs[None, :]]
    V = flow_np[1, ys[:, None], xs[None, :]]
    plt.imshow(rgb)
    plt.quiver(X, Y, U, V, color='r', angles='xy', scale_units='xy', scale=1.0, width=0.003)
    plt.axis('off')

# New: Save GT/Pred/Diff three images
def save_flows_figure(flow_gt: torch.Tensor, flow_pred: torch.Tensor, sample_idx: int, out_dir: str):
    """Save GT, Pred, and Diff optical flow visualizations separately, instead of merging into one image.
    Output files:
      - sample_XXXXX_flow_gt.png
      - sample_XXXXX_flow_pred.png
      - sample_XXXXX_flow_diff.png
    """
    os.makedirs(out_dir, exist_ok=True)
    flow_gt_img = flow_viz.flow_to_image(flow_gt.permute(1, 2, 0).numpy())
    flow_pred_img = flow_viz.flow_to_image(flow_pred.permute(1, 2, 0).numpy())
    flow_diff_img = flow_viz.flow_to_image((flow_pred - flow_gt).permute(1, 2, 0).numpy())

    items = [
        (f'sample_{sample_idx:05d}_flow_gt.png', flow_gt_img),
        (f'sample_{sample_idx:05d}_flow_pred.png', flow_pred_img),
        (f'sample_{sample_idx:05d}_flow_diff.png', flow_diff_img),
    ]
    for filename, img in items:
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img / 255.0)
        ax.axis('off')
        fig.tight_layout()
        fig_path = os.path.join(out_dir, filename)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

# New: Save arrow image
def save_arrows_figure(image_t: torch.Tensor, flow_pred: torch.Tensor, sample_idx: int, out_dir: str, step: int = 32):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(5, 5), dpi=150)
    show_flow_arrow(image_t, flow_pred, step=step)
    fig_path = os.path.join(out_dir, f'sample_{sample_idx:05d}_arrows.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

def save_arrows_warped_gt(image_t: torch.Tensor, flow_gt: torch.Tensor, sample_idx: int, out_dir: str, step: int = 32):
    """On the image warped by GT optical flow, draw sparse arrows in the reverse direction of flow_gt (-flow_gt) and save.
    Input:
      - image_t: Tensor of [1,1,H,W] or [1,C,H,W] (recommended to pass grayscale 1 channel)
      - flow_gt: GT optical flow of [2,H,W] (matching spatial dimensions of image_t)
    Output: sample_XXXXX_arrows_warp_gt_rev.png
    """
    os.makedirs(out_dir, exist_ok=True)
    # Ensure tensors are on the same device
    device = image_t.device
    img_b = image_t.to(device)
    flow_b = flow_gt.unsqueeze(0).to(device)  # [1,2,H,W]
    # Use GT optical flow to warp the image to align with image2
    try:
        img_warp = flow_viz.get_warp_flow(img_b, flow_b).detach().cpu()
    except Exception:
        # Fallback: if get_warp_flow is unavailable, use the original image directly (still drawing reverse arrows)
        img_warp = img_b.detach().cpu()
    # Draw reverse direction arrows (-flow_gt) on the warped image
    fig = plt.figure(figsize=(5, 5), dpi=150)
    show_flow_arrow(img_warp, (-flow_gt).detach().cpu(), step=step)
    fig_path = os.path.join(out_dir, f'sample_{sample_idx:05d}_arrows_warp_gt_rev.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

# New: Save metrics JSON
def save_metrics_json(epe_np: np.ndarray, checkpoint: str, sample_idx: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        'index': int(sample_idx),
        'checkpoint': checkpoint,
        'epe_mean': float(epe_np.mean()),
        'px1': float(np.mean(epe_np < 1)),
        'px3': float(np.mean(epe_np < 3)),
        'px5': float(np.mean(epe_np < 5))
    }
    with open(os.path.join(out_dir, f'sample_{sample_idx:05d}_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

def _tensor_to_display_rgb(tensor_img: torch.Tensor) -> np.ndarray:
    """Convert tensor image to uint8 RGB numpy array for plotting."""
    if tensor_img.dim() == 4:
        arr = tensor_img[0].cpu().numpy()
    elif tensor_img.dim() == 3:
        arr = tensor_img.cpu().numpy()
    else:
        arr = tensor_img.unsqueeze(0).cpu().numpy()

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)

    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = (arr * 255.0).round()
    return arr.astype(np.uint8)


def show_pic_with_polygon(tensor_img: torch.Tensor, poly_ref: np.ndarray, poly_other: np.ndarray, subplot_idx: int, total: int):
    rgb = _tensor_to_display_rgb(tensor_img)
    ax = plt.subplot(1, total, subplot_idx)
    ax.imshow(rgb)
    # Swap colors: poly_ref uses red, poly_other uses yellow
    if poly_ref is not None:
        poly = np.vstack([poly_ref, poly_ref[0]])
        ax.plot(poly[:, 0], poly[:, 1], color='red', linewidth=2)
    if poly_other is not None:
        poly = np.vstack([poly_other, poly_other[0]])
        ax.plot(poly[:, 0], poly[:, 1], color='yellow', linewidth=2)
    ax.axis('off')


def save_polygon_figure(image_opt: torch.Tensor,
                        image_sar: torch.Tensor,
                        flow_gt: torch.Tensor,
                        flow_pred: torch.Tensor,
                        sample_idx: int,
                        out_dir: str,
                        box_ratio: float = 0.6,
                        affine_step: int = 8):
    """Save four images separately:
      1) Original optical image + center box (yellow) -> sample_XXXXX_opt.png
      2) Optical image warped by GT affine + box (yellow=GT) -> sample_XXXXX_opt_warp_gt.png
      3) SAR image + box (yellow=GT) -> sample_XXXXX_sar.png
      4) Optical image warped by Pred affine + box (yellow=GT, red=Pred) -> sample_XXXXX_opt_warp_pred.png
    No longer merged into a single four-panel image.
    """
    os.makedirs(out_dir, exist_ok=True)
    image_opt = image_opt.cpu()
    image_sar = image_sar.cpu()
    flow_gt = flow_gt.cpu()
    flow_pred = flow_pred.cpu()

    H, W = flow_gt.shape[1], flow_gt.shape[2]
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid spatial dimensions for flow_gt, H and W must be greater than 0, but got H={H}, W={W}")
    
    ratio = np.clip(box_ratio, 0.05, 0.99)
    inner_w = int(W * ratio)
    inner_h = int(H * ratio)
    left = int((W - inner_w) / 2)
    top = int((H - inner_h) / 2)
    right = left + inner_w - 1
    bottom = top + inner_h - 1
    pts_corners = np.array([[left, top], [left, bottom], [right, bottom], [right, top]], dtype=np.float32)

    H_gt = estimate_affine_from_flow(flow_gt, step=affine_step)

    if H_gt.shape != (2, 3):
        raise ValueError(f"H_gt format error, expected [2, 3], but got {H_gt.shape}")

    H_pred = estimate_affine_from_flow(flow_pred, step=affine_step)

    if H_gt.shape != (2, 3):
        raise ValueError(f"H_gt format error, expected [2, 3], but got {H_gt.shape}")
    if H_pred.shape != (2, 3):
        raise ValueError(f"H_pred format error, expected [2, 3], but got {H_pred.shape}")

    pts_gt = PoinsTransf(pts_corners, H_gt)
    pts_pred = PoinsTransf(pts_corners, H_pred)

    # Process optical image
    if image_opt.dim() == 4:  # [B, C, H, W]
        if image_opt.shape[1] == 3:  # RGB image
            opt_tensor = image_opt[0]  # Keep RGB format
        elif image_opt.shape[1] == 1:  # Grayscale image
            opt_tensor = image_opt[0, 0:1]  # Keep grayscale format
        else:
            raise ValueError(f"image_opt channel error, expected 1 or 3 channels, but got {image_opt.shape[1]} channels")
    elif image_opt.dim() == 3:  # [C, H, W]
        if image_opt.shape[0] == 3:  # RGB image
            opt_tensor = image_opt  # Keep RGB format
        elif image_opt.shape[0] == 1:  # Grayscale image
            opt_tensor = image_opt[0:1]  # Keep grayscale format
        else:
            raise ValueError(f"image_opt channel error, expected 1 or 3 channels, but got {image_opt.shape[0]} channels")
    else:
        raise ValueError(f"image_opt dimension error, expected 3 or 4 dim tensor, but got {image_opt.dim()} dim")

    # Process SAR image
    if image_sar.dim() == 4:  # [B, C, H, W]
        if image_sar.shape[1] == 3:  # RGB image
            sar_tensor = image_sar[0]  # Keep RGB format
        elif image_sar.shape[1] == 1:  # Grayscale image
            sar_tensor = image_sar[0, 0:1]  # Keep grayscale format
        else:
            raise ValueError(f"image_sar channel error, expected 1 or 3 channels, but got {image_sar.shape[1]} channels")
    elif image_sar.dim() == 3:  # [C, H, W]
        if image_sar.shape[0] == 3:  # RGB image
            sar_tensor = image_sar  # Keep RGB format
        elif image_sar.shape[0] == 1:  # Grayscale image
            sar_tensor = image_sar[0:1]  # Keep grayscale format
        else:
            raise ValueError(f"image_sar channel error, expected 1 or 3 channels, but got {image_sar.shape[0]} channels")
    else:
        raise ValueError(f"image_sar dimension error, expected 3 or 4 dim tensor, but got {image_sar.dim()} dim")

    # Convert optical image to grayscale for affine transformation
    if opt_tensor.shape[0] == 3:  # RGB to grayscale
        opt_gray = opt_tensor[0:1] * 0.2989 + opt_tensor[1:2] * 0.5870 + opt_tensor[2:3] * 0.1140
    else:  # Grayscale image
        opt_gray = opt_tensor

    opt_gray_np = opt_gray.squeeze(0).numpy().astype(np.float32)

    if opt_gray_np.ndim != 2:  # Ensure it is a 2D image
        raise ValueError(f"opt_gray_np dimension error, expected 2D array, but got {opt_gray_np.shape}")

    if opt_gray_np.shape[0] <= 0 or opt_gray_np.shape[1] <= 0:
        raise ValueError(f"opt_gray_np size error, width and height must be greater than zero, but got {opt_gray_np.shape}")

    warp_gt_np = cv2.warpAffine(opt_gray_np, H_gt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    warp_pred_np = cv2.warpAffine(opt_gray_np, H_pred, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    
    if image_opt.dim() == 4:  # [B, C, H, W]
        if image_opt.shape[1] == 3:  # RGB image
            opt_gray_np_r = opt_tensor[0].numpy().astype(np.float32)
            opt_gray_np_g = opt_tensor[1].numpy().astype(np.float32)
            opt_gray_np_b = opt_tensor[2].numpy().astype(np.float32)

            warp_gt_np_r = cv2.warpAffine(opt_gray_np_r, H_gt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warp_gt_np_g = cv2.warpAffine(opt_gray_np_g, H_gt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warp_gt_np_b = cv2.warpAffine(opt_gray_np_b, H_gt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            warp_pred_np_r = cv2.warpAffine(opt_gray_np_r, H_pred, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warp_pred_np_g = cv2.warpAffine(opt_gray_np_g, H_pred, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warp_pred_np_b = cv2.warpAffine(opt_gray_np_b, H_pred, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            warp_gt_tensor = torch.stack([
                torch.from_numpy(warp_gt_np_r),
                torch.from_numpy(warp_gt_np_g),
                torch.from_numpy(warp_gt_np_b)
            ], dim=0).unsqueeze(0)

            warp_pred_tensor = torch.stack([
                torch.from_numpy(warp_pred_np_r),
                torch.from_numpy(warp_pred_np_g),
                torch.from_numpy(warp_pred_np_b)
            ], dim=0).unsqueeze(0)
    elif image_opt.dim() == 3 and image_opt.shape[0] == 3:  # [C, H, W] RGB image
        opt_gray_np_r = opt_tensor[0].numpy().astype(np.float32)
        opt_gray_np_g = opt_tensor[1].numpy().astype(np.float32)
        opt_gray_np_b = opt_tensor[2].numpy().astype(np.float32)

        warp_gt_np_r = cv2.warpAffine(opt_gray_np_r, H_gt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warp_gt_np_g = cv2.warpAffine(opt_gray_np_g, H_gt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warp_gt_np_b = cv2.warpAffine(opt_gray_np_b, H_gt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        warp_pred_np_r = cv2.warpAffine(opt_gray_np_r, H_pred, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warp_pred_np_g = cv2.warpAffine(opt_gray_np_g, H_pred, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warp_pred_np_b = cv2.warpAffine(opt_gray_np_b, H_pred, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        warp_gt_tensor = torch.stack([
            torch.from_numpy(warp_gt_np_r),
            torch.from_numpy(warp_gt_np_g),
            torch.from_numpy(warp_gt_np_b)
        ], dim=0).unsqueeze(0)

        warp_pred_tensor = torch.stack([
            torch.from_numpy(warp_pred_np_r),
            torch.from_numpy(warp_pred_np_g),
            torch.from_numpy(warp_pred_np_b)
        ], dim=0).unsqueeze(0)
    else:  # If grayscale image
        warp_gt_tensor = torch.from_numpy(warp_gt_np).unsqueeze(0).unsqueeze(0)
        warp_pred_tensor = torch.from_numpy(warp_pred_np).unsqueeze(0).unsqueeze(0)

    def _center_crop_rgb(rgb_img: np.ndarray, target: int = 512) -> np.ndarray:
        """Crop the RGB image to target x target around the center without padding."""
        h, w = rgb_img.shape[:2]
        crop_h = min(target, h)
        crop_w = min(target, w)
        top = max((h - crop_h) // 2, 0)
        left = max((w - crop_w) // 2, 0)
        bottom = top + crop_h
        right = left + crop_w
        return rgb_img[top:bottom, left:right]
    
    def _save_single(img_t: torch.Tensor, poly_ref: np.ndarray, poly_other: np.ndarray, path: str):
        rgb = _center_crop_rgb(_to_color_u8(img_t))
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(rgb)
        # Swap colors: poly_ref -> red, poly_other -> yellow
        if poly_ref is not None:
            poly = np.vstack([poly_ref, poly_ref[0]])
            ax.plot(poly[:, 0], poly[:, 1], color='red', linewidth=2)
        if poly_other is not None:
            poly = np.vstack([poly_other, poly_other[0]])
            ax.plot(poly[:, 0], poly[:, 1], color='yellow', linewidth=2)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    _save_single(opt_tensor, pts_corners, None, os.path.join(out_dir, f'sample_{sample_idx:05d}_opt.png'))
    _save_single(warp_gt_tensor, pts_gt, None, os.path.join(out_dir, f'sample_{sample_idx:05d}_opt_warp_gt.png'))
    _save_single(sar_tensor, pts_gt, None, os.path.join(out_dir, f'sample_{sample_idx:05d}_sar.png'))
    _save_single(warp_pred_tensor, pts_gt, pts_pred, os.path.join(out_dir, f'sample_{sample_idx:05d}_opt_warp_pred.png'))

def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = -flo
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    import matplotlib.pyplot as plt
    plt.imshow(img_flo / 255.0)
    plt.show()


    

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24, polygon_dir=None, box_ratio=0.6, affine_step=8):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='testing')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

        # Visualize and save prediction results for each pair (restructured path: separate folder per sample)
        try:
            padder = InputPadder(image1.shape)
            img0 = padder.unpad(image1[0]).cpu()
            img1 = padder.unpad(image2[0]).cpu()
            flow_full = padder.unpad(flow_pr[0]).cpu()

            data = {
                'image0': img0.unsqueeze(0),
                'image1': img1.unsqueeze(0),
                'flow_f_full': flow_full.unsqueeze(0),
                'flow_gt_full': flow_gt.unsqueeze(0)
            }

            avg_epe = float(epe.mean().item()) if epe.numel() > 0 else -1.0
            ret_dict = {'metrics': {'AEPE': [avg_epe]}}

            out_dir = os.path.join('eval_figs', 'chairs')
            os.makedirs(out_dir, exist_ok=True)
            fig = _make_evaluation_figure(data, 0, ret_dict=ret_dict)
            fig_path = os.path.join(out_dir, f'chairs_{val_id:05d}.png')
            fig.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)

            if polygon_dir:
                sample_dir = os.path.join(polygon_dir, f'sample_{val_id:05d}')
                os.makedirs(sample_dir, exist_ok=True)
                save_polygon_figure(
                    img0,
                    img1,
                    flow_gt,
                    flow_full,
                    val_id,
                    sample_dir,
                    box_ratio=box_ratio,
                    affine_step=affine_step,
                )
                save_flows_figure(flow_gt, flow_full, val_id, sample_dir)
                img0_gray = img0[0:1].unsqueeze(0) if img0.dim() == 3 else img0.unsqueeze(0)
                save_arrows_figure(img0_gray, flow_full, val_id, sample_dir, step=32)
                save_metrics_json(epe.view(-1).numpy(), '', val_id, sample_dir)
        except Exception:
            pass

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=12, args=None):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    
    for dstype in [args.dstype] if args is not None else [args.dstype]:
        
        val_dataset = datasets.loadOS_90rot_dataset(split='testing', dstype=args.dstype) 
        epe_list = []

        # New path structure: out_dir/dstype/sample_xxxxxx/*
        base_dstype_dir = os.path.join(args.out_dir, dstype)
        if args is not None:
            os.makedirs(base_dstype_dir, exist_ok=True)

        total = len(val_dataset)
        limit = total if (args.max_samples is None or args.max_samples < 0) else min(args.max_samples, total)
        for val_id in range(limit):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1_p, image2_p = padder.pad(image1, image2)

            _, flow_pr = model(image1_p, image2_p, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()


            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            
            print("img_id: %.f, EPE: %.3f" % (val_id, epe.mean().item())) 
      

            # Save each image to sample exclusive directory
            try:
                # Unpad to get original size image/flow
                img0 = padder.unpad(image1_p[0]).cpu()
                img1 = padder.unpad(image2_p[0]).cpu()
                flow_full = flow

                # Design hierarchical save path: out_dir/dstype/sample_xxxxx/{polygons,flows,arrows,metrics,evaluation}
                sample_root = os.path.join(base_dstype_dir, f'sample_{val_id:05d}')
                sub_polygons = os.path.join(sample_root, 'polygons')
                sub_flows = os.path.join(sample_root, 'flows')
                sub_arrows = os.path.join(sample_root, 'arrows')
                sub_metrics = os.path.join(sample_root, 'metrics')
                sub_eval = os.path.join(sample_root, 'evaluation')
                for d in [sub_polygons, sub_flows, sub_arrows, sub_metrics, sub_eval]:
                    os.makedirs(d, exist_ok=True)

                # 1) Polygon four images
                save_polygon_figure(
                    img0,
                    img1,
                    flow_gt,
                    flow_full,
                    val_id,
                    sub_polygons,
                    box_ratio=(args.box_ratio if args is not None else 0.6),
                    affine_step=(args.affine_sample_step if args is not None else 8),
                )
                # 2) Flow three images
                save_flows_figure(flow_gt, flow_full, val_id, sub_flows)
                # 3) Sparse arrow image
                img0_gray = img0[0:1].unsqueeze(0) if img0.dim() == 3 else img0.unsqueeze(0)
                # img1_gray = img1[0:1].unsqueeze(0) if img1.dim() == 3 else img1.unsqueeze(0)
                save_arrows_figure(img0_gray, flow_full, val_id, sub_arrows, step=32)
                # 3.1) Draw reverse arrow (-flow_gt) on GT-warp image
                # save_arrows_warped_gt(img1_gray, flow_gt, val_id, sub_arrows, step=32)
                # # 4) Metrics JSON
                # save_metrics_json(epe.view(-1).numpy(), args.model if args is not None else '', val_id, sub_metrics)

                # 5) Use _make_evaluation_figure to generate connection visualization, and save/optionally display
                data = {
                    'image0': img0.unsqueeze(0),
                    'image1': img1.unsqueeze(0),
                    'flow_f_full': flow_full.unsqueeze(0),
                    'flow_gt_full': flow_gt.unsqueeze(0),
                }
                avg_epe = float(epe.mean().item()) if epe.numel() > 0 else -1.0
                ret_dict = {'metrics': {'AEPE': [avg_epe]}}
                fig_eval = _make_evaluation_figure(data, 0, ret_dict=ret_dict)
                eval_path = os.path.join(sub_eval, 'evaluation.png')
                fig_eval.savefig(eval_path, bbox_inches='tight')

                # Display one by one (optional)
                if args is not None and getattr(args, 'show', False):
                    try:
                        # Non-blocking display, control stay time by show_pause; <=0 blocks until manually closed
                        if args.show_pause and args.show_pause > 0:
                            plt.show(block=False)
                            plt.pause(args.show_pause)
                            plt.close(fig_eval)
                        else:
                            plt.show(block=True)
                    except Exception:
                        plt.close(fig_eval)
                else:
                    plt.close(fig_eval)
            except Exception:
                # Avoid single sample error interrupting the entire evaluation
                try:
                    plt.close('all')
                except Exception:
                    pass
                pass

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px2 = np.mean(epe_all < 2)
        px3 = np.mean(epe_all < 3)
        px4 = np.mean(epe_all < 4)
        px5 = np.mean(epe_all < 5)
        px1p = 100 * np.sum(epe_all < 1) / epe_all.size
        px2p = 100 * np.sum(epe_all < 2) / epe_all.size
        px3p = 100 * np.sum(epe_all < 3) / epe_all.size
        px4p = 100 * np.sum(epe_all < 4) / epe_all.size
        px5p = 100 * np.sum(epe_all < 5) / epe_all.size
        
        px01 = np.mean(epe_all < 0.1)
        px02 = np.mean(epe_all < 0.2)
        px03 = np.mean(epe_all < 0.3)
        px04 = np.mean(epe_all < 0.4)
        px05 = np.mean(epe_all < 0.5)
        px06 = np.mean(epe_all < 0.6)
        px07 = np.mean(epe_all < 0.7)
        px08 = np.mean(epe_all < 0.8)
        px09 = np.mean(epe_all < 0.9)
        px01p = 100 * np.sum(epe_all < 0.1) / epe_all.size
        px02p = 100 * np.sum(epe_all < 0.2) / epe_all.size
        px03p = 100 * np.sum(epe_all < 0.3) / epe_all.size
        px04p = 100 * np.sum(epe_all < 0.4) / epe_all.size
        px05p = 100 * np.sum(epe_all < 0.5) / epe_all.size
        px06p = 100 * np.sum(epe_all < 0.6) / epe_all.size
        px07p = 100 * np.sum(epe_all < 0.7) / epe_all.size
        px08p = 100 * np.sum(epe_all < 0.8) / epe_all.size
        px09p = 100 * np.sum(epe_all < 0.9) / epe_all.size
        
        print("EPE: %.4f, 0.1px: %.3f(%.1f%%), 0.2px: %.3f(%.1f%%), 0.3px: %.3f(%.1f%%), 0.4px: %.3f(%.1f%%), 0.5px: %.3f(%.1f%%), 0.6px: %.3f(%.1f%%), 0.7px: %.3f(%.1f%%), 0.8px: %.3f(%.1f%%), 0.9px: %.3f(%.1f%%)" % (epe, px01, px01p, px02, px02p, px03, px03p, px04, px04p, px05, px05p, px06, px06p, px07, px07p, px08, px08p, px09, px09p))
        print("1px: %.3f(%.1f%%), 2px: %.3f(%.1f%%), 3px: %.3f(%.1f%%), 4px: %.3f(%.1f%%), 5px: %.3f(%.1f%%)" % (px1, px1p, px2, px2p, px3, px3p, px4, px4p, px5, px5p))
        # print("Validation (%s) EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (dstype_ori, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results

def _to_color_u8(image_t: torch.Tensor) -> np.ndarray:
    """Convert tensor to 8-bit RGB color image (H,W,3). Supports [C,H,W]/[1,C,H,W], C=1/3."""
    if image_t.dim() == 4:
        arr = image_t[0].detach().cpu().numpy()  # [C,H,W]
    elif image_t.dim() == 3:
        arr = image_t.detach().cpu().numpy()
    else:  # [H,W]
        arr = image_t.unsqueeze(0).detach().cpu().numpy()

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # (H,W,C)
    elif arr.ndim == 2:
        arr = arr[:, :, None]

    # If single channel, repeat to 3 channels
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = (arr * 255.0).round()
    return arr.astype(np.uint8)

@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='testing')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


#figure
def _make_evaluation_figure(data, b_id, alpha='dynamic', ret_dict=None):
    # Keep original image colors, do not grayscale or sharpen
    def _tensor_or_array_to_hwc(x):
        # x can be a torch.Tensor or numpy array
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # possible shapes: (C,H,W), (H,W,C), (H,W)
        if x.ndim == 3 and x.shape[0] in (1, 3):
            # (C,H,W) -> (H,W,C)
            x = np.transpose(x, (1, 2, 0))

        if x.ndim == 3 and x.shape[2] == 1:
            # (H,W,1) -> (H,W,3)
            x = np.concatenate([x] * 3, axis=2)

        if x.ndim == 2:
            # (H,W) -> (H,W,3)
            x = np.stack([x] * 3, axis=-1)

        # Convert to uint8 if needed. Handle floats in [0,1].
        if x.dtype != np.uint8:
            x = x.astype(np.float32)
            if x.max() <= 1.0:
                x = (x * 255.0).round().astype(np.uint8)
            else:
                x = x.round().astype(np.uint8)

        return x

    img0 = _tensor_or_array_to_hwc(data['image0'][b_id])
    img1 = _tensor_or_array_to_hwc(data['image1'][b_id])

    H, W, _ = img0.shape
    
    combined_img = np.concatenate([img0, img1], axis=1)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.imshow(combined_img)
    ax.axis('off')

    pts0, pts1_pred = None, None
    avg_epe = -1
    
    if ret_dict and 'metrics' in ret_dict and 'AEPE' in ret_dict['metrics']:
        if b_id < len(ret_dict['metrics']['AEPE']):
            avg_epe = ret_dict['metrics']['AEPE'][b_id]

    # Visualize using keypoints
    # if 'mkpts0_f' in data and 'mkpts1_f' in data:
    #     pts0 = data['mkpts0_f'][b_id].cpu().numpy()
    #     pts1_pred = data['mkpts1_f'][b_id].cpu().numpy()
        

    # Visualization based on corner point sampling
    if 'flow_f_full' in data:
        flow_pred = data['flow_f_full'][b_id].cpu()  # [2, H, W]

        if flow_pred.shape[-2:] != (H, W):
            flow_pred = F.interpolate(flow_pred.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)

        # Use Harris corner detection to select meaningful points for visualization
        img0_gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(
            img0_gray,
            maxCorners=1000,       # Max display x corners
            qualityLevel=0.01,    # Corner quality threshold
            minDistance=1        # Min distance between corners
        )

        if corners is not None and len(corners) > 0:
            pts0 = np.squeeze(corners, axis=1) # [N, 2]
        else: # If no corners detected, fallback to sparse grid sampling
            step = 5
            y_coords, x_coords = np.mgrid[step//2:H:step, step//2:W:step]
            pts0 = np.stack((x_coords.ravel(), y_coords.ravel()), axis=-1)

        pts0_int = pts0.astype(int)
        # Crop coordinates to prevent out of bounds
        pts0_int[:, 0] = np.clip(pts0_int[:, 0], 0, W - 1)
        pts0_int[:, 1] = np.clip(pts0_int[:, 1], 0, H - 1)

        # Get displacement of predicted flow at sampled points
        flow_pred_pts = flow_pred[:, pts0_int[:, 1], pts0_int[:, 0]].T.numpy()
        pts1_pred = pts0 + flow_pred_pts

        # If GT flow provided, use it to filter wrong matches (distance > 5px considered wrong)
        flow_gt = None
        pts1_gt = None
        if 'flow_gt_full' in data and data['flow_gt_full'] is not None:
            flow_gt = data['flow_gt_full'][b_id].cpu()
            if flow_gt.shape[-2:] != (H, W):
                flow_gt = F.interpolate(flow_gt.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)

            flow_gt_pts = flow_gt[:, pts0_int[:, 1], pts0_int[:, 0]].T.numpy()
            pts1_gt = pts0 + flow_gt_pts
        
    if pts0 is not None and pts1_pred is not None:
        valid_pts0 = []
        valid_pts1_pred = []
        margin = 5
        # Collect all valid match points (filter out of bounds, black pixels, edge points, and distance threshold based on GT flow)
        for i in range(len(pts0)):
            pt0 = pts0[i]
            pt1 = pts1_pred[i]

            # Check if pt1 is within the valid area of img1
            pt1_int = pt1.astype(int)
            if not (0 <= pt1_int[1] < H and 0 <= pt1_int[0] < W):
                continue  # Skip out of bounds points

            # Edge filtering: both pt0 and pt1 must be further than margin from edge
            if not (margin <= pt0[0] < W - margin and margin <= pt0[1] < H - margin):
                continue
            if not (margin <= pt1[0] < W - margin and margin <= pt1[1] < H - margin):
                continue

            # Skip if target point is completely black (possible invalid area)
            if np.all(img1[pt1_int[1], pt1_int[0]] == 0):
                continue

            # If GT flow exists, verify prediction against GT: keep only matches with error <= 5 pixels
            if pts1_gt is not None:
                pt1_gt = pts1_gt[i]
                dist = np.linalg.norm(pt1 - pt1_gt)
                if dist > 5.0:
                    continue

            valid_pts0.append(pt0)
            valid_pts1_pred.append(pt1)

        # Draw all connections and keypoints with unified neon green
        n = len(valid_pts0)
        if n > 0:
            green_color = (0.0, 1.0, 0.0, 0.95)  # RGBA Neon Green
            for j in range(n):
                pt0 = valid_pts0[j]
                pt1 = valid_pts1_pred[j]

                # Start circle (original image side)
                ax.add_artist(plt.Circle(pt0, radius=0.5, color=green_color, fill=False, linewidth=1))
                # End circle (target image side, slightly larger radius 0.9)
                ax.add_artist(plt.Circle((pt1[0] + W, pt1[1]), radius=0.9, color=green_color, fill=False, linewidth=1))

                # Connection line (unified neon green, transparency from color alpha)
                line = plt.Line2D((pt0[0], pt1[0] + W), (pt0[1], pt1[1]),
                                  linewidth=1, color=green_color, alpha=green_color[3])
                ax.add_artist(line)

            if avg_epe >= 0:
                text = f'AEPE = {avg_epe:.2f}px'
            ax.text(0.5, -0.05, text, ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, "No matching data available",
                    color='white', fontsize=14, ha='center', va='center',
                    bbox=dict(facecolor='red', alpha=0.7))

    else:
        ax.text(0.5, 0.5, "No matching data available", 
                color='white', fontsize=14, ha='center', va='center',
                bbox=dict(facecolor='red', alpha=0.7))

    plt.tight_layout()
    return fig



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', default="./checkpoints/raft-pre-30平移0.1尺度10旋转contrary.pth", help="restore checkpoint")
    # parser.add_argument('--model', default="./checkpoints/10000_raft-pre-UBCv2dataset_szxcleanwithval512_30平移0.1尺度10旋转_crop256.pth", help="restore checkpoint")
    parser.add_argument('--model', default="./checkpoints/100000_raft-pre-UBCv2dataset_szxcleanwithval512_30平移0.1尺度10旋转_crop256.pth", help="restore checkpoint")

    
    parser.add_argument('--dstype', default="30平移0.1尺度10旋转2", type=str)
    parser.add_argument('--dataset', default="sintel", help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256])

    # New: Output and visualization related parameters
    parser.add_argument('--out_dir', type=str, default='fig_4points', help='Output root directory (automatically subdivided into polygons/flows/arrows/metrics/evaluation)')
    parser.add_argument('--box_ratio', type=float, default=0.6, help='Center box relative width/height ratio (0-1)')
    parser.add_argument('--affine_sample_step', type=int, default=8, help='Sampling step for estimating affine matrix')
    parser.add_argument('--max_samples', type=int, default=-1, help='Process only first N samples, -1 means all')
    # Display related parameters
    parser.add_argument('--show', action='store_true', help='Display visualization window one by one (evaluation image)')
    parser.add_argument('--show_pause', type=float, default=0.7, help='Display time per image (seconds), <=0 blocks until window closed manually')

    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            # Pass args to enable saving
            validate_sintel(model.module, args=args)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)


