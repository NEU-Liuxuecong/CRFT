import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
from einops.einops import rearrange
import torch.nn.functional as F
import torch
import cv2
import matplotlib.colors  


def flow_to_color(flow, max_flow=None):
    """
    Convert optical flow to color coding, similar to GMFlow visualization.
    
    Args:
        flow: [H, W, 2] or [2, H, W] optical flow.
        max_flow: Maximum flow value for normalization.
    
    Returns:
        color_image: [H, W, 3] RGB image.
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()
    
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)  # [2, H, W] -> [H, W, 2]
    
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    
    if max_flow is None:
        max_flow = np.max(magnitude)
    
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi) * 255  # Hue
    hsv[:, :, 1] = 255  # Saturation
    hsv[:, :, 2] = np.clip(magnitude / max_flow * 255, 0, 255)  # Value
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def make_flow_visualization(img0, img1, flow_pred, flow_gt=None, text=None, dpi=75):
    """
    Create optical flow visualization, similar to GMFlow style.
    
    Args:
        img0, img1: Input images [H, W] or [H, W, 3].
        flow_pred: Predicted optical flow [2, H, W] or [H, W, 2].
        flow_gt: Ground truth optical flow [2, H, W] or [H, W, 2], optional.
        text: Text information to display.
        dpi: Image DPI.
    
    Returns:
        matplotlib figure
    """
    if flow_gt is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=dpi)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
        axes = [axes]
    
    # Ensure image is 3-channel
    if len(img0.shape) == 2:
        img0 = np.stack([img0] * 3, axis=-1)
    if len(img1.shape) == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    
    # First row: Input images and predicted flow
    axes[0][0].imshow(img0.astype(np.uint8))
    axes[0][0].set_title('Image 0')
    axes[0][0].axis('off')
    
    axes[0][1].imshow(img1.astype(np.uint8))
    axes[0][1].set_title('Image 1')
    axes[0][1].axis('off')
    
    # Predicted flow visualization
    flow_pred_color = flow_to_color(flow_pred)
    axes[0][2].imshow(flow_pred_color)
    axes[0][2].set_title('Predicted Flow')
    axes[0][2].axis('off')
    
    if flow_gt is not None:
        # Second row: Ground truth flow, error map, and flow difference
        flow_gt_color = flow_to_color(flow_gt)
        axes[1][0].imshow(flow_gt_color)
        axes[1][0].set_title('Ground Truth Flow')
        axes[1][0].axis('off')
        
        # Calculate flow error
        if isinstance(flow_pred, torch.Tensor):
            flow_pred_np = flow_pred.cpu().numpy()
        else:
            flow_pred_np = flow_pred
            
        if isinstance(flow_gt, torch.Tensor):
            flow_gt_np = flow_gt.cpu().numpy()
        else:
            flow_gt_np = flow_gt
        
        # Ensure shapes match
        if flow_pred_np.shape != flow_gt_np.shape:
            if flow_pred_np.ndim == 3 and flow_pred_np.shape[0] == 2:
                flow_pred_np = flow_pred_np.transpose(1, 2, 0)
            if flow_gt_np.ndim == 3 and flow_gt_np.shape[0] == 2:
                flow_gt_np = flow_gt_np.transpose(1, 2, 0)
        
        error = np.sqrt(np.sum((flow_pred_np - flow_gt_np)**2, axis=-1))
        
        # Error heatmap
        im = axes[1][1].imshow(error, cmap='hot', vmin=0, vmax=np.percentile(error, 95))
        axes[1][1].set_title('Flow Error (EPE)')
        axes[1][1].axis('off')
        plt.colorbar(im, ax=axes[1][1], fraction=0.046, pad=0.04)
        
        # Flow difference visualization
        flow_diff = flow_pred_np - flow_gt_np
        flow_diff_color = flow_to_color(flow_diff, max_flow=np.percentile(np.sqrt(np.sum(flow_diff**2, axis=-1)), 95))
        axes[1][2].imshow(flow_diff_color)
        axes[1][2].set_title('Flow Difference')
        axes[1][2].axis('off')
    
    # Add text information
    if text is not None:
        text_str = '\n'.join(text)
        fig.text(0.02, 0.98, text_str, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def process_image(img_tensor):
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
    
    if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):  # Grayscale
        if img_np.ndim == 3:
            img_np = img_np.squeeze(axis=2)
        return np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 3:  # RGB
        return img_np  # Return RGB directly, do not convert to grayscale
    else:  
        # For other channel counts (e.g., 4), take the first channel as grayscale
        img_gray = img_np[:, :, 0]
        return np.stack([img_gray] * 3, axis=-1)


def _make_evaluation_figure(data, b_id, alpha='dynamic', ret_dict=None):
    """
    Create evaluation figure - visualize optical flow or keypoints using matching lines.
    """
    img0 = data['image0'][b_id].cpu()
    img1 = data['image1'][b_id].cpu()

    img0 = process_image(img0)
    img1 = process_image(img1)

    H_orig, W_orig, _ = img0.shape

    if 'flow_f_full' in data and data['flow_f_full'] is not None:
        H_proc, W_proc = data['flow_f_full'][b_id].shape[1:]
    else:
        H_proc, W_proc = data['image0_model_shape']

    scale_h = H_orig / H_proc
    scale_w = W_orig / W_proc
    # Concatenate two images horizontally
    combined_img = np.concatenate([img0, img1], axis=1)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.imshow(combined_img)
    ax.axis('off')

    pts0, pts1_pred, pts1_gt = None, None, None
    avg_epe = -1
    epe_list = []

    # Get pre-computed MAE as AEPE from ret_dict
    if ret_dict and 'metrics' in ret_dict and 'AEPE' in ret_dict['metrics']:
        if b_id < len(ret_dict['metrics']['AEPE']):
            avg_epe = ret_dict['metrics']['AEPE'][b_id]

    # Use corner-based sampling for visualization if flow data exists
    if 'flow_f_full' in data:
        flow_pred = data['flow_f_full'][b_id].cpu()  # [2, H, W]

        # Use Harris corner detection on original image to select meaningful points (denser)
        img0_gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)  # Use original image
        corners = cv2.goodFeaturesToTrack(
            img0_gray, maxCorners=5000, qualityLevel=0.0001, minDistance=0.5)

        if corners is not None and len(corners) > 0:
            pts0 = np.squeeze(corners, axis=1)  # [N, 2] in original resolution
            # Map to processing resolution to get flow
            pts0_proc = pts0 / np.array([scale_w, scale_h])
            pts0_proc_int = pts0_proc.astype(int)
            pts0_proc_int[:, 0] = np.clip(pts0_proc_int[:, 0], 0, W_proc - 1)
            pts0_proc_int[:, 1] = np.clip(pts0_proc_int[:, 1], 0, H_proc - 1)
        else:  # Fallback to sparse grid sampling if no corners detected
            step = 3
            y_coords, x_coords = np.mgrid[step//2:H_orig:step, step//2:W_orig:step]
            pts0 = np.stack((x_coords.ravel(), y_coords.ravel()), axis=-1)
            pts0_proc = pts0 / np.array([scale_w, scale_h])
            pts0_proc_int = pts0_proc.astype(int)
            pts0_proc_int[:, 0] = np.clip(pts0_proc_int[:, 0], 0, W_proc - 1)
            pts0_proc_int[:, 1] = np.clip(pts0_proc_int[:, 1], 0, H_proc - 1)

        # Get corresponding points from predicted flow
        flow_pred_pts = flow_pred[:, pts0_proc_int[:, 1], pts0_proc_int[:, 0]].T.numpy()
        flow_pred_pts_scaled = flow_pred_pts * np.array([scale_w, scale_h])
        pts1_pred = pts0 + flow_pred_pts_scaled
        
        # Calculate EPE (in low resolution)
        if 'flow' in data:
            flow_gt = data['flow'][b_id].cpu()  # [2, H_proc, W_proc]
            flow_gt_pts = flow_gt[:, pts0_proc_int[:, 1], pts0_proc_int[:, 0]].T.numpy()
            pts1_gt_proc = pts0_proc + flow_gt_pts  # GT corresponding points in low res
            pts1_pred_proc = pts0_proc + flow_pred_pts  # Predicted corresponding points in low res
            epe_list = np.linalg.norm(pts1_pred_proc - pts1_gt_proc, axis=1)  # Calculate EPE in low res
            if avg_epe < 0:
                 avg_epe = np.mean(epe_list) if len(epe_list) > 0 else -1

    if pts0 is not None and pts1_pred is not None:
        # Draw matching lines (keep only points with EPE < 5)
        for i in range(len(pts0)):
            # Check EPE < 5 if list available, otherwise draw all
            if len(epe_list) > i and epe_list[i] >= 5: # Threshold is 5px as per original logic comment
                continue  # Skip points with EPE >= 5
            
            pt0 = pts0[i]
            pt1 = pts1_pred[i]
            if not (0 <= pt0[0] < W_orig and 0 <= pt0[1] < H_orig):
                continue  
            if not (0 <= pt1[0] < W_orig and 0 <= pt1[1] < H_orig):
                continue  
            
            green_color = (0.0, 1.0, 0.0, 0.95)
            ax.add_artist(plt.Circle(pt0, radius=0.5, color=green_color, fill=False, linewidth=1))
            ax.add_artist(plt.Circle((pt1[0] + W_orig, pt1[1]), radius=0.9, color=green_color, fill=False, linewidth=1))
                        
            line = plt.Line2D((pt0[0], pt1[0] + W_orig), (pt0[1], pt1[1]),
                              linewidth=1, color=green_color, alpha=0.8)
            ax.add_artist(line)

        if avg_epe >= 0:
            text = f'AEPE = {avg_epe:.2f}px'
            ax.text(0.5, -0.05, text, ha='center', va='center', transform=ax.transAxes, fontsize=12)

    else:
        ax.text(0.5, 0.5, "No matching data available", 
                color='white', fontsize=14, ha='center', va='center',
                bbox=dict(facecolor='red', alpha=0.7))

    plt.tight_layout()
    return fig


def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure for flow
    raise NotImplementedError("Confidence visualization for flow not implemented")


def make_matching_figures(data, config, mode='evaluation', ret_dict=None):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_CRFT.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA, ret_dict=ret_dict)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)
