import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import os.path as osp
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
    print('Fallback to latest weights:', osp.basename(pths[0]))
    return pths[0]


def PoinsTransf(pts: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Apply affine matrix h (2x3) to point set pts[[x,y],...].
    """
    # Check input data
    if pts.shape[1] != 2:
        raise ValueError(f"Point set format error, expected shape [N,2], but got {pts.shape}")
    if h.shape != (2, 3):
        raise ValueError(f"Affine matrix format error, expected shape [2,3], but got {h.shape}")

    # Apply affine transformation
    pts_h = np.concatenate([pts.astype(np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)  # [N,3]
    out = (h.astype(np.float32) @ pts_h.T).T  # [N,2]

    # Debug info
    print(f"Affine transformation complete. Input points: {pts.shape}, Output points: {out.shape}")
    return out


def PoinsTransf_flow(pts: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Displace corner points using pixel-level optical flow (2,H,W) by directly sampling corresponding pixel displacements.
    """
    # Check input data
    if flow.shape[0] != 2:
        raise ValueError(f"Optical flow format error, expected shape [2,H,W], but got {flow.shape}")
    if pts.shape[1] != 2:
        raise ValueError(f"Point set format error, expected shape [N,2], but got {pts.shape}")

    # Apply flow displacement
    H, W = flow.shape[1], flow.shape[2]
    pts_int = pts.astype(int)
    pts_int[:, 0] = np.clip(pts_int[:, 0], 0, W - 1)
    pts_int[:, 1] = np.clip(pts_int[:, 1], 0, H - 1)
    disp = flow[:, pts_int[:, 1], pts_int[:, 0]].T  # [N,2]
    out = pts.astype(np.float32) + disp.astype(np.float32)

    # Debug info
    print(f"Flow displacement complete. Input points: {pts.shape}, Output points: {out.shape}")
    return out


def estimate_affine_from_flow(flow: torch.Tensor, step: int = 8) -> np.ndarray:
    """
    Estimate least squares affine matrix H (2x3) from dense optical flow.
    """
    # Check input data
    if isinstance(flow, torch.Tensor):
        flow_np = flow.detach().cpu().numpy()
    else:
        flow_np = np.asarray(flow, dtype=np.float32)

    if flow_np.shape[0] != 2:
        raise ValueError(f"Optical flow format error, expected shape [2,H,W], but got {flow_np.shape}")

    # Sample optical flow
    H_, W_ = flow_np.shape[1], flow_np.shape[2]
    xs, ys = np.meshgrid(np.arange(W_, dtype=np.float32), np.arange(H_, dtype=np.float32))

    xs_sample = xs[::step, ::step].reshape(-1)
    ys_sample = ys[::step, ::step].reshape(-1)
    u_sample = flow_np[0, ::step, ::step].reshape(-1)
    v_sample = flow_np[1, ::step, ::step].reshape(-1)

    # Ensure sufficient sample size
    if xs_sample.size < 6:
        xs_sample = xs.reshape(-1)
        ys_sample = ys.reshape(-1)
        u_sample = flow_np[0].reshape(-1)
        v_sample = flow_np[1].reshape(-1)

    # Least squares solution for affine parameters
    ones = np.ones_like(xs_sample)
    A = np.stack([xs_sample, ys_sample, ones], axis=1)  # [N,3]
    bx = xs_sample + u_sample
    by = ys_sample + v_sample

    px, *_ = np.linalg.lstsq(A, bx, rcond=None)
    py, *_ = np.linalg.lstsq(A, by, rcond=None)

    H_mat = np.stack([px, py], axis=0).astype(np.float32)  # [2,3]

    # Debug info
    print(f"Affine matrix estimation complete. Matrix: {H_mat}")
    return H_mat


def _to_display_image(tensor_img: torch.Tensor) -> np.ndarray:
    """
    Convert tensor image to displayable RGB image (uint8 format).
    Supports grayscale and RGB images.
    """
    if tensor_img.dim() == 4:
        arr = tensor_img[0].cpu().numpy()  # [C, H, W]
    elif tensor_img.dim() == 3:
        arr = tensor_img.cpu().numpy()  # [C, H, W]
    else:
        arr = tensor_img.unsqueeze(0).cpu().numpy()  # [1, H, W]

    if arr.shape[0] == 1:  # Grayscale image
        arr = np.repeat(arr, 3, axis=0)  # Convert to pseudo-RGB
    arr = np.transpose(arr, (1, 2, 0))  # [C, H, W] -> [H, W, C]

    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = (arr * 255.0).round()
    return arr.astype(np.uint8)


def save_flow_visualizations(image1, image2, flow_pred, flow_gt, out_dir, sample_idx):
    """
    Save optical flow visualizations, including flow map, arrow map, and checkerboard map.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert to displayable image
    img1_disp = _to_display_image(image1)
    img2_disp = _to_display_image(image2)

    # Optical flow visualization
    flow_pred_img = flow_viz.flow_to_image(flow_pred.permute(1, 2, 0).cpu().numpy())
    flow_gt_img = flow_viz.flow_to_image(flow_gt.permute(1, 2, 0).cpu().numpy())

    # Save flow map (Commented out in original)
    #cv2.imwrite(osp.join(out_dir, f'sample_{sample_idx:05d}_flow_pred.png'), cv2.cvtColor(flow_pred_img, cv2.COLOR_RGB2BGR))
    #cv2.imwrite(osp.join(out_dir, f'sample_{sample_idx:05d}_flow_gt.png'), cv2.cvtColor(flow_gt_img, cv2.COLOR_RGB2BGR))

    # Save arrow map
    fig = plt.figure(figsize=(5, 5), dpi=150)
    show_flow_arrow(image1, flow_pred, step=32)
    plt.savefig(osp.join(out_dir, f'sample_{sample_idx:05d}_arrows.png'), bbox_inches='tight')
    plt.close(fig)

    # Save checkerboard map
    save_checkerboard_image(image1, image2, flow_pred, out_dir, base_name=f'sample_{sample_idx:05d}')


def process_image(img_tensor):
    """
    Process input image tensor, ensuring compatibility with grayscale and RGB images.
    """
    # If input is 4D tensor [B, C, H, W], remove batch dimension
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]  # Remove batch dimension

    min_val = img_tensor.min()
    max_val = img_tensor.max()

    # If max value > 1, assume 0-255 range, do not normalize
    if max_val > 1:
        img_np = img_tensor.numpy().astype(np.uint8)
    else:
        # Normalize to 0-255
        if max_val > min_val:
            img_tensor = (img_tensor - min_val) / (max_val - min_val)
        img_np = (img_tensor.numpy() * 255).round().astype(np.uint8)

    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))  # [C, H, W] -> [H, W, C]

    # Ensure output is RGB format
    if img_np.ndim == 2:  # Grayscale
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.ndim == 3 and img_np.shape[2] == 1:  # Single channel image
        img_np = np.repeat(img_np, 3, axis=2)
    elif img_np.ndim == 3 and img_np.shape[2] != 3:  # Non-RGB image
        raise ValueError(f"Input image channel error, expected 1 or 3 channels, but got {img_np.shape[2]} channels")

    return img_np


def save_checkerboard_image(image1, image2, flow_pred, out_dir, base_name='sample', checkersize=65):
    """
    Save checkerboard comparison image showing alignment between image1 and image2_warp.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Check input image dimensions
    if image1.dim() not in [3, 4]:
        raise ValueError(f"image1 dimension error, expected 3D or 4D tensor, but got {image1.dim()}D tensor")
    if image2.dim() not in [3, 4]:
        raise ValueError(f"image2 dimension error, expected 3D or 4D tensor, but got {image2.dim()}D tensor")
    # Warp image2 using predicted flow
    H, W = int(flow_pred.shape[1]), int(flow_pred.shape[2])
    device = image1.device
    flow_pred = flow_pred.unsqueeze(0).to(device)  # [1, 2, H, W]
    image2_warp = get_warp_flow(image2.unsqueeze(0).to(device), -flow_pred)
    image1_warp = get_warp_flow(image1.unsqueeze(0).to(device), flow_pred)
    # Convert to displayable images
    flow_pred_cpu = flow_pred.detach().cpu()
    H_pred = estimate_affine_from_flow(flow_pred_cpu.squeeze(0), step=8)
    H_pred_inv = cv2.invertAffineTransform(H_pred)
    x1_affine = _warp_tensor_affine(image1, H_pred, (H, W))
    x2_affine = _warp_tensor_affine(image2, H_pred_inv, (H, W))

    def _save_checker(a_tensor: torch.Tensor, b_tensor: torch.Tensor, suffix: str):
        img_a = _to_color_u8(a_tensor.detach().cpu())
        img_b = _to_color_u8(b_tensor.detach().cpu())
        H, W, _ = img_a.shape
        ys, xs = np.indices((H, W))
        mask = ((ys // checkersize) + (xs // checkersize)) % 2
        mask3 = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8)
        comp = img_a * (1 - mask3) + img_b * mask3
        cv2.imwrite(
            os.path.join(out_dir, f'{base_name}_checkerboard_{suffix}.png'),
            cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR),
        )

    _save_checker(image1, image2_warp, 'img1_img2warp')
    _save_checker(image1_warp, image2, 'img1warp_img2')
    _save_checker(x1_affine, image2, 'img1affine_img2')
    _save_checker(image1, x2_affine, 'img1_img2affine')

def show_flow_arrow(image_t, flow_t, step=32):
    """
    Draw sparse arrows on the image to represent optical flow.
    """
    img = _to_display_image(image_t)
    flow_np = flow_t.cpu().numpy()
    H, W = flow_np.shape[1], flow_np.shape[2]
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    X, Y = np.meshgrid(xs, ys)
    U = flow_np[0, ys[:, None], xs[None, :]]
    V = flow_np[1, ys[:, None], xs[None, :]]

    plt.imshow(img)
    plt.quiver(X, Y, U, V, color='r', angles='xy', scale_units='xy', scale=1.0, width=0.003)
    plt.axis('off')
def _warp_tensor_affine(image_t: torch.Tensor,
                        H_aff: np.ndarray,
                        out_hw: tuple[int, int],
                        border_mode: int = cv2.BORDER_CONSTANT,
                        border_value: float = 0) -> torch.Tensor:
    """Apply affine warp to a tensor image (single sample) channel by channel."""
    image_cpu = image_t.detach().cpu()
    if image_cpu.dim() == 4:
        image_cpu = image_cpu[0]

    H_out, W_out = out_hw
    if image_cpu.dim() == 3:
        warped = []
        for c in range(image_cpu.shape[0]):
            channel = image_cpu[c].numpy().astype(np.float32)
            warped.append(
                cv2.warpAffine(
                    channel,
                    H_aff,
                    (W_out, H_out),
                    flags=cv2.INTER_LINEAR,
                    borderMode=border_mode,
                    borderValue=border_value,
                )
            )
        warped_np = np.stack(warped, axis=0)
        return torch.from_numpy(warped_np).unsqueeze(0)
    elif image_cpu.dim() == 2:
        channel = image_cpu.numpy().astype(np.float32)
        warped_np = cv2.warpAffine(
            channel,
            H_aff,
            (W_out, H_out),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=border_value,
        )
        return torch.from_numpy(warped_np).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Expected tensor with shape [C,H,W] or [H,W]")


def show_pic_with_polygon(tensor_img: torch.Tensor, poly_ref: np.ndarray, poly_other: np.ndarray, subplot_idx: int, total: int):
    """Grayscale display + two polygons overlay. tensor_img: [1,1,H,W] or [1,C,H,W]."""
    if tensor_img.dim() == 4:
        # Use first channel for display
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
    # Swapped colors: poly_ref to yellow, poly_other to red
    if poly_ref is not None:
        cv2.polylines(rgb, [poly_ref.astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
    if poly_other is not None:
        cv2.polylines(rgb, [poly_other.astype(np.int32)], True, (255, 0, 0), 2, cv2.LINE_AA)
    ax = plt.subplot(1, total, subplot_idx)
    ax.imshow(rgb)
    ax.axis('off')


def PoinsTransf(pts: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply affine matrix h (2x3) to point array pts[[x,y], ...]."""
    pts = np.asarray(pts, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    out = (h @ pts_h.T).T
    return out


def estimate_affine_from_flow(flow: torch.Tensor, step: int = 8) -> np.ndarray:
    """
    Estimate least squares affine matrix H (2x3) from dense optical flow.
    """
    # Check input data
    if isinstance(flow, torch.Tensor):
        flow_np = flow.detach().cpu().numpy()
    else:
        flow_np = np.asarray(flow, dtype=np.float32)

    if flow_np.shape[0] != 2:
        raise ValueError(f"Optical flow format error, expected shape [2,H,W], but got {flow_np.shape}")

    # Sample optical flow
    H_, W_ = flow_np.shape[1], flow_np.shape[2]
    xs, ys = np.meshgrid(np.arange(W_, dtype=np.float32), np.arange(H_, dtype=np.float32))

    xs_sample = xs[::step, ::step].reshape(-1)
    ys_sample = ys[::step, ::step].reshape(-1)
    u_sample = flow_np[0, ::step, ::step].reshape(-1)
    v_sample = flow_np[1, ::step, ::step].reshape(-1)

    # Ensure sufficient sample size
    if xs_sample.size < 6:
        xs_sample = xs.reshape(-1)
        ys_sample = ys.reshape(-1)
        u_sample = flow_np[0].reshape(-1)
        v_sample = flow_np[1].reshape(-1)

    # Least squares solution for affine parameters
    ones = np.ones_like(xs_sample)
    A = np.stack([xs_sample, ys_sample, ones], axis=1)  # [N,3]
    bx = xs_sample + u_sample
    by = ys_sample + v_sample

    px, *_ = np.linalg.lstsq(A, bx, rcond=None)
    py, *_ = np.linalg.lstsq(A, by, rcond=None)

    H_mat = np.stack([px, py], axis=0).astype(np.float32)  # [2,3]

    # Debug info
    print(f"Affine matrix estimation complete. Matrix: {H_mat}")
    return H_mat

# === Key warp functions copied from gmutils/flow_viz.py (self-contained) ===
def get_grid(batch_size, H, W, start=0):
    if torch.cuda.is_available():
        xx = torch.arange(0, W).cuda()
        yy = torch.arange(0, H).cuda()
    else:
        xx = torch.arange(0, W)
        yy = torch.arange(0, H)
    xx = xx.view(1, -1).repeat(H, 1)
    yy = yy.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    ones = torch.ones_like(xx).cuda() if torch.cuda.is_available() else torch.ones_like(xx)
    grid = torch.cat((xx, yy, ones), 1).float()
    grid[:, :2, :, :] = grid[:, :2, :, :] + start
    return grid

def transformer(I, vgrid, train=True):
    def _interpolate(im, x, y, out_size):
        """
        Bilinear interpolation function, supports 4D tensor input.
        """
        # Ensure input is 4D tensor
        if im.dim() == 3:  # [C, H, W]
            im = im.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
        elif im.dim() != 4:
            raise ValueError(f"Input tensor dimension error, expected 4D or 3D tensor, but got {im.dim()}D tensor")

        num_batch, num_channels, height, width = im.size()
        out_height, out_width = out_size[0], out_size[1]
        zero = 0
        max_y = height - 1
        max_x = width - 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.clamp(x0, 0, width - 1)
        x1 = torch.clamp(x1, 0, width - 1)
        y0 = torch.clamp(y0, 0, height - 1)
        y1 = torch.clamp(y1, 0, height - 1)
        dim1 = width * height
        dim2 = width
        if torch.cuda.is_available():
            base = torch.arange(0, num_batch).int().cuda()
        else:
            base = torch.arange(0, num_batch).int()
        base = base * dim1
        base = base.repeat_interleave(out_height * out_width, axis=0)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, num_channels]).float()
        idx_a = idx_a.unsqueeze(-1).long().expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)
        idx_b = idx_b.unsqueeze(-1).long().expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)
        idx_c = idx_c.unsqueeze(-1).long().expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)
        idx_d = idx_d.unsqueeze(-1).long().expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)
        x0_f = x0.float(); x1_f = x1.float(); y0_f = y0.float(); y1_f = y1.float()
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    def _transform(I, vgrid):
        C_img = I.shape[1]
        B, C, H, W = vgrid.size()
        x_s_flat = vgrid[:, 0, ...].reshape([-1])
        y_s_flat = vgrid[:, 1, ...].reshape([-1])
        out_size = vgrid.shape[2:]
        input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size)
        output = input_transformed.reshape([B, H, W, C_img])
        return output

    output = _transform(I, vgrid)
    if train:
        output = output.permute(0, 3, 1, 2)
    return output

def get_warp_flow(img, flow, start=0):
    """
    Warp image using optical flow.
    """
    # Check input image dimensions
    if img.dim() == 5:  # If 5D tensor, try to remove extra dimension
        img = img.squeeze(0)  # Remove first dimension
    if img.dim() == 3:  # [C, H, W]
        img = img.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
    elif img.dim() != 4:
        raise ValueError(f"Input image dimension error, expected 4D or 3D tensor, but got {img.dim()}D tensor")

    batch_size, _, patch_size_h, patch_size_w = flow.shape
    grid_warp = get_grid(batch_size, patch_size_h, patch_size_w, start)[:, :2, :, :] - flow
    img_warp = transformer(img, grid_warp)
    flow = torch.clamp(flow, min=-patch_size_w, max=patch_size_w)
    return img_warp

# Added: Sparse arrow optical flow
def show_flow_arrow(image_t: torch.Tensor, flow_t: torch.Tensor, step: int = 32):
    """Sparse arrow representation of optical flow. image_t: [1,C,H,W] flow_t: [2,H,W]"""
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

# Added: Save GT/Pred/Diff three separate images
def save_flows_figure(flow_gt: torch.Tensor, flow_pred: torch.Tensor, sample_idx: int, out_dir: str):
    """Save GT, Pred, Diff optical flow visualizations separately, not merged.
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

# Added: Save arrow map
def save_arrows_figure(image_t: torch.Tensor, flow_pred: torch.Tensor, sample_idx: int, out_dir: str, step: int = 32):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(5, 5), dpi=150)
    show_flow_arrow(image_t, flow_pred, step=step)
    fig_path = os.path.join(out_dir, f'sample_{sample_idx:05d}_arrows.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

def save_arrows_warped_gt(image_t: torch.Tensor, flow_gt: torch.Tensor, sample_idx: int, out_dir: str, step: int = 32):
    """Draw reverse GT flow arrows (-flow_gt) on the image warped by GT flow and save.
    Inputs:
      - image_t: [1,1,H,W] or [1,C,H,W] tensor (recommend passing grayscale 1 channel)
      - flow_gt: [2,H,W] GT optical flow (matches image_t spatial size)
    Output: sample_XXXXX_arrows_warp_gt_rev.png
    """
    os.makedirs(out_dir, exist_ok=True)
    # Ensure tensors are on the same device
    device = image_t.device
    img_b = image_t.to(device)
    flow_b = flow_gt.unsqueeze(0).to(device)  # [1,2,H,W]
    # Warp image using GT flow to get image aligned with image2
    try:
        # Use get_warp_flow within this file to avoid external dependency issues
        img_warp = get_warp_flow(img_b, flow_b).detach().cpu()
    except Exception:
        # Fallback: if get_warp_flow is unavailable, use original image (still draw reverse arrows)
        img_warp = img_b.detach().cpu()
    # Draw reverse direction arrows (-flow_gt) on the warped image
    fig = plt.figure(figsize=(5, 5), dpi=150)
    show_flow_arrow(img_warp, (-flow_gt).detach().cpu(), step=step)
    fig_path = os.path.join(out_dir, f'sample_{sample_idx:05d}_arrows_warp_gt_rev.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

# Added: Save Metrics JSON
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

# === Save flow_viz bundle output: img1/img1_warp/img2/img2_warp/flow_gt/flow_pred ===
def _to_gray_u8(image_t: torch.Tensor) -> np.ndarray:
    """Convert [1,1,H,W] or [1,C,H,W]/[C,H,W] tensor to 8-bit grayscale 2D array [H,W]."""
    if image_t.dim() == 4:
        arr = image_t[0, 0].detach().cpu().numpy()
    elif image_t.dim() == 3:
        # If color C=3, reduce to grayscale using channel 0; maintain original interface semantics
        arr = image_t[0].detach().cpu().numpy()
    else:
        arr = image_t.detach().cpu().numpy()
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = (arr * 255.0).round()
    return arr.astype(np.uint8)

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


def save_flowviz_bundle(image1: torch.Tensor,
                        image2: torch.Tensor,
                        flow_gt: torch.Tensor,
                        flow_pred: torch.Tensor,
                        out_dir: str,
                        sample_idx: int,
                        base_name: str = 'sample'):
    """Save flow_viz bundle:
    - img1.png, img1_warp.png, img2.png, img2_warp.png, flow_gt.png, flow_pred.png
    Convention:
    - image1/image2 tensors are [C,H,W] (CPU), channel 0 will be used for grayscale if needed.
    - flow_* are [2,H,W] (CPU).
    - warp uses flow_pred: img1_warp = warp(img1, flow_pred), img2_warp = warp(img2, -flow_pred)
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W = int(flow_pred.shape[1]), int(flow_pred.shape[2])
    H_pred = estimate_affine_from_flow(flow_pred, step=8)

    # Unify device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Ensure input tensors are float
    image1 = image1.to(device).float()
    image2 = image2.to(device).float()
    flow_gt = flow_gt.to(device).float()
    flow_pred = flow_pred.to(device).float()
    
    # Calculate flow difference
    flow_diff = flow_pred - flow_gt
    zero_flow = torch.zeros_like(flow_pred)  # [2, H, W]

    # Warp image2 using predicted flow
    if image2.dim()==3: x2 = image2.unsqueeze(0).to(device)
    else: x2 = image2.unsqueeze(0).unsqueeze(0).to(device)
    flow_b = flow_pred.unsqueeze(0).to(device)
    try:
        x2_warp = get_warp_flow(x2, -flow_b)
    except Exception:
        x2_warp = x2.clone()
    
    img1 = _to_color_u8(image1.detach().cpu())
    img2 = _to_color_u8(image2.detach().cpu())
    img1_warp_tensor = _warp_tensor_affine(image1, H_pred, (H, W))
    img1_warp = _to_color_u8(img1_warp_tensor)
    img2_warp = _to_color_u8(x2_warp.detach().cpu())
    
    # flow visualization in color
    flow_gt_img = flow_viz.flow_to_image(flow_gt.permute(1, 2, 0).cpu().numpy())
    flow_pred_img = flow_viz.flow_to_image(flow_pred.permute(1, 2, 0).cpu().numpy())
    flow_diff_img = flow_viz.flow_to_image(flow_diff.permute(1, 2, 0).cpu().numpy())

    # Save results
    cv2.imwrite(os.path.join(out_dir,f'{base_name}_img1.png'), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir,f'{base_name}_img1_warp.png'), cv2.cvtColor(img1_warp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir,f'{base_name}_img2.png'), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir,f'{base_name}_img2_warp.png'), cv2.cvtColor(img2_warp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_{sample_idx:05d}_flow_gt.png'), cv2.cvtColor(flow_gt_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_{sample_idx:05d}_flow_pred.png'), cv2.cvtColor(flow_pred_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_{sample_idx:05d}_flow_diff.png'), cv2.cvtColor(flow_diff_img, cv2.COLOR_RGB2BGR))


