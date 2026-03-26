from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRFTLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # global config
        self.loss_config = config['crft']['loss']
        # Weights (unused in current logic but kept for config compatibility if needed)
        # self.pos_w = self.loss_config['pos_weight']
        # self.neg_w = self.loss_config['neg_weight']

    def compute_fine_matching_loss(self, data):
        """
        Compute fine-level flow MAE loss.
        Directly calculates the MAE loss between predicted flow and ground truth flow.
        
        Args:
            data (dict): Dictionary containing:
                - 'flow_f_full': (B, 2, H, W) Predicted fine-level optical flow.
                - 'flow': (B, 2, H, W) Ground truth optical flow.
                - 'mask': (B, H, W) Optional mask for valid regions.
        
        Returns:
            torch.Tensor: The calculated loss scalar.
        """
        # Check if flow output exists
        if 'flow_f_full' not in data or 'flow' not in data:
            return torch.tensor(0.0, device=data.get('image0', torch.device('cpu')).device, requires_grad=True)
        
        flow_pred = data['flow_f_full']  # (B, 2, H, W)
        flow_gt = data['flow']           # (B, 2, H, W)
        
        # Ensure dimensions match
        if flow_pred.shape != flow_gt.shape:
            flow_pred = F.interpolate(
                flow_pred, 
                size=flow_gt.shape[-2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # Calculate MAE loss
        flow_diff = torch.abs(flow_pred - flow_gt)  # (B, 2, H, W)
        
        # Apply mask if exists
        if 'mask' in data:
            mask = data['mask'].float()  # (B, H, W)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, H, W)
            
            # Ensure mask shape matches flow_diff
            if mask.shape[-2:] != flow_diff.shape[-2:]:
                 mask = F.interpolate(mask, size=flow_diff.shape[-2:], mode='nearest')
                 
            flow_diff = flow_diff * mask
            loss = flow_diff.sum() / (mask.sum() * 2 + 1e-8)  # Divide by 2 because of x, y components
        else:
            loss = flow_diff.mean()
        
        return loss
    
    def compute_coarse_loss(self, data):
        """
        Compute coarse-level flow MAE loss.
        
        Args:
            data (dict): Dictionary containing:
                - 'flow_c': (B, 2, H_c, W_c) Predicted coarse-level optical flow.
                  OR 'flow_f_full': (B, 2, H, W) Fallback to fine flow if coarse is missing.
                - 'flow': (B, 2, H, W) Ground truth optical flow.
                - 'mask': (B, H, W) Optional mask for valid regions.
        
        Returns:
            torch.Tensor: The calculated loss scalar.
        """
        # Prioritize using coarse-level flow output
        if 'flow_c' in data:
            flow_pred = data['flow_c']  # (B, 2, H_c, W_c)
        elif 'flow_f_full' in data:
            # If no coarse output, downsample fine output
            flow_pred = data['flow_f_full']  # (B, 2, H, W)
            # Downsample to coarse resolution (assuming stride 2 based on original logic)
            flow_pred = F.avg_pool2d(flow_pred, kernel_size=2, stride=2)
        else:
            return torch.tensor(0.0, device=data.get('image0', torch.device('cpu')).device, requires_grad=True)
        
        flow_gt = data['flow']  # (B, 2, H, W)
        
        # Downsample ground truth flow to match predicted flow resolution
        if flow_pred.shape[-2:] != flow_gt.shape[-2:]:
            scale_factor = flow_gt.shape[-2] // flow_pred.shape[-2]
            if scale_factor > 0:
                # Downsample GT flow to coarse resolution
                flow_gt_coarse = F.avg_pool2d(flow_gt, 
                                            kernel_size=scale_factor,
                                            stride=scale_factor)
            else:
                 flow_gt_coarse = flow_gt
        else:
            flow_gt_coarse = flow_gt
        
        # Calculate MAE loss
        flow_diff = torch.abs(flow_pred - flow_gt_coarse)  # (B, 2, H_c, W_c)
        
        # Apply mask if exists (downsample mask to match coarse resolution)
        if 'mask' in data:
            mask = data['mask'].float()  # (B, H, W)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, H, W)
            
            if mask.shape[-2:] != flow_pred.shape[-2:]:
                scale_factor = mask.shape[-2] // flow_pred.shape[-2]
                mask = F.avg_pool2d(mask, 
                                  kernel_size=scale_factor,
                                  stride=scale_factor)
            
            flow_diff = flow_diff * mask
            loss = flow_diff.sum() / (mask.sum() * 2 + 1e-8)  # Divide by 2 because of x, y components
        else:
            loss = flow_diff.mean()
        
        return loss

    def forward(self, data):
        """
        Forward pass using MAE loss for both coarse and fine levels.
        
        Updates `data` dict with:
            - 'loss': (torch.Tensor) The scalar total loss.
            - 'loss_scalars': (dict) Loss components for logging.
        """
        loss_scalars = {}
        
        # 1. Coarse-level MAE loss
        loss_c = self.compute_coarse_loss(data)
        if self.loss_config.get('coarse_weight', 0):
             loss_c = loss_c * self.loss_config['coarse_weight'] 
        loss = loss_c 
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. Fine-level MAE loss
        loss_f = self.compute_fine_matching_loss(data)
        if self.loss_config.get('fine_weight', 0):
            loss_f = loss_f * self.loss_config['fine_weight'] 
        loss = loss + loss_f 
        loss_scalars.update({"loss_f": loss_f.clone().detach().cpu()})

        # Total loss
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
