import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from src.crft.utils.geometry import coords_grid
from torch.utils import checkpoint as cp

INF = 1e9

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int): border length
        v (m.dtype): value to fill
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_flow(flow, b: int):
    """
    Args:
        flow (torch.Tensor): [B, 2, H, W] flow field
    """
    if b <= 0:
        return flow
    
    flow[:, :, :b] = 0      # Top border
    flow[:, :, -b:] = 0     # Bottom border  
    flow[:, :, :, :b] = 0   # Left border
    flow[:, :, :, -b:] = 0  # Right border
    
    return flow


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    """Mask borders with padding"""
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    # Case for 2D mask
    if p_m0.dim() == 2:
        # Single mask case, convert to batch dimension
        h0s = torch.tensor([p_m0.sum(1).max(-1)[0].int()], device=p_m0.device)
        w0s = torch.tensor([p_m0.sum(-1).max(-1)[0].int()], device=p_m0.device)
        h1s = torch.tensor([p_m1.sum(1).max(-1)[0].int()], device=p_m1.device)
        w1s = torch.tensor([p_m1.sum(-1).max(-1)[0].int()], device=p_m1.device)
    else:
        # Multiple masks case
        h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
        h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


def checkpointed(module, *args):
    # return cp.checkpoint(lambda *x: module(*x), *args)
    return module(*args)

class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        d_model = config['d_model']
        self.thr = config['thr']
        self.inference = config['inference']
        self.border_rm = config['border_rm']
        # -- # for training fine-level CRFT
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']
        self.final_proj = nn.Linear(d_model, d_model, bias=True)

        self.temperature = config['dsmax_temperature']

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'flow_c' (torch.Tensor): [B, 2, H, W]
            }
        """

        feat_c0 = self.final_proj(feat_c0)
        feat_c1 = self.final_proj(feat_c1)

        B = feat_c0.shape[0]
        h0, w0 = data['hw0_c']
        h1, w1 = data['hw1_c']
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])
        feat_c0_reshaped = feat_c0.permute(0, 2, 1).reshape(B, -1, h0, w0)
        feat_c1_reshaped = feat_c1.permute(0, 2, 1).reshape(B, -1, h1, w1)

        flow_c, prob = global_correlation_softmax(feat_c0_reshaped, feat_c1_reshaped)
        
        # flow border processing
        flow_c = mask_border_flow(flow_c, self.border_rm)
        
        data['flow_c'] = flow_c  # 1/8 resolution flow
        if 'hw0_i' in data:
            h0_i, w0_i = data['hw0_i']
            h1_i, w1_i = data['hw1_i']
            flow_c_up = upsample_flow(flow_c, (h0_i, w0_i))
        return
        
def upsample_flow(flow, target_size, scale_factor=None):
    if scale_factor is None:
        scale_factor = target_size[0] / flow.shape[2]
        
    flow_up = F.interpolate(flow, size=target_size, 
                           mode='bilinear', align_corners=False)
    return flow_up * scale_factor

def global_correlation_softmax(feature0, feature1, pred_bidir_flow=False):
    # global correlation
    b, c, h, w = feature0.shape 
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C] 
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]

    # Matrix multiplication to compute similarity
    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)  # [B, H, W, H, W]

    # flow from softmax
    init_grid = coords_grid(b, h, w).to(correlation.device)  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]

    if pred_bidir_flow:
        correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)  # [2*B, H*W, H*W]
        init_grid = init_grid.repeat(2, 1, 1, 1)  # [2*B, 2, H, W]
        grid = grid.repeat(2, 1, 1)  # [2*B, H*W, 2]
        b = b * 2

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W] 
    correspondence = torch.matmul(prob, grid.to(prob.dtype)).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid

    return flow, prob
