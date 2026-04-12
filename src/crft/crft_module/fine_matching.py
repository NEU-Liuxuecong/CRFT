import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
class LocalFlowRefinement(nn.Module):
    def __init__(self, dim, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        
        # Local feature extraction
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        
        # Flow refinement network
        self.flow_refine = nn.Sequential(
            nn.Conv2d( 2, 16, 3, padding=1),  
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, 3, padding=1)
        )
        self.feature_flow_attn = FeatureFlowAttention(in_channels=dim)
    def forward(self, feat0, feat1, flow_init):
        # Local feature extraction
        local_feat0 = self.local_conv(feat0)
        local_feat1 = self.local_conv(feat1)
        
        # Warp feat1 using initial flow
        warped_feat1 = self.warp_features(local_feat1, flow_init)
        
        # Calculate feature difference
        feat_diff = local_feat0 - warped_feat1
        
        # Concatenate feature difference and initial flow
        #flow_input = torch.cat([feat_diff, flow_init], dim=1)#[B,C,H,W] + [B,2,H,W] = [B,C+2,H,W]
        feat_diff = F.normalize(feat_diff, p=2, dim=1)
        flow_input = self.feature_flow_attn(1-feat_diff, flow_init,
                                          True,
                                          local_window_radius=2)
        # Refine flow
        flow_delta = self.flow_refine(flow_input)
        
        return flow_init + flow_delta
    
    def warp_features(self, feat, flow):
        B, C, H, W = feat.shape
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device),
            torch.arange(W, device=feat.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        #Find corresponding position by adding flow to grid
        warped_grid = grid + flow
        
        # Normalize to [-1, 1]
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        
        # Rearrange to grid_sample format [B, H, W, 2]
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Execute warp
        warped_feat = F.grid_sample(feat, warped_grid, mode='bilinear', 
                                   padding_mode='border', align_corners=True) #Sample and rotate
        
        return warped_feat

class FlowConfidenceEstimator(nn.Module):
    """Flow confidence estimator"""
    def __init__(self, dim):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Conv2d(dim * 2 + 2, dim, 3, padding=1),  # +2 for flow
            nn.GELU(),
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, feat0, feat1, flow):
        # Warp feat1 using flow
        warped_feat1 = self.warp_features(feat1, flow)
        
        # Calculate feature similarity
        feat_concat = torch.cat([feat0, warped_feat1, flow], dim=1)
        confidence = self.confidence_net(feat_concat)
        
        return confidence
    
    def warp_features(self, feat, flow):
        B, C, H, W = feat.shape
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device),
            torch.arange(W, device=feat.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        warped_grid = grid + flow
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        return F.grid_sample(feat, warped_grid, mode='bilinear', 
                           padding_mode='border', align_corners=True)

class FineMatching(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim_f = config['resnet']['block_dims'][0]  # 128
        self.dim = dim_f
        
        # Flow estimation parameters
        self.num_iterations = 10  
        
        # Feature projection layer
        self.feat_proj = nn.Sequential(
            nn.Linear(dim_f, dim_f),
            nn.GELU(),
            nn.Linear(dim_f, dim_f)
        )
        
        # Local flow refinement module
        self.flow_refinement = LocalFlowRefinement(dim_f)
        
        # Flow confidence estimator
        self.confidence_estimator = FlowConfidenceEstimator(dim_f)
        
        # Edge-aware smoother
        self.edge_aware_smoother = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, 3, padding=1)
        )
        
       
    def forward(self, feat_f0_unfold, feat_f1_unfold, data):
        """Main flow estimation function"""
       
        # Process input feature dimensions
        feat_f0, feat_f1, is_windowed = self._prepare_features(feat_f0_unfold, feat_f1_unfold, data)
        
        flow_field, confidence,iter_flow_field,iter_confidence = self._compute_windowed_flow(feat_f0, feat_f1, data) #''' '''
        # Edge-aware smoothing
        flow_field = self._edge_aware_smoothing(flow_field, confidence)
        
        # Upsample to full image resolution
        flow_full = self._upsample_to_image_resolution(flow_field, data)
        for i in range(self.num_iterations):
            iter_flow_field[i] = self._edge_aware_smoothing(iter_flow_field[i], iter_confidence[i])
            iter_flow_field[i]=self._upsample_to_image_resolution(iter_flow_field[i], data)
        
        # Update data dictionary
        data.update({
            'flow_f': flow_field,           # Feature-level flow field
            'flow_f_full': flow_full,       # Image-level flow field  
            'flow_confidence': confidence,   # Flow confidence
            'iter_flow_f': iter_flow_field,      
            'num_iterations':self.num_iterations
        })
     
    def _prepare_features(self, feat_f0_unfold, feat_f1_unfold, data):
        """Prepare and process input features - compute flow separately in each window"""
        
        # Process different input formats
        if len(feat_f0_unfold.shape) == 4:  # [B, L, WW, C] - Window feature format
            B, L, WW, C = feat_f0_unfold.shape
            
            # Reshape window features to flat format: [B, L, WW, C] -> [B, L*WW, C]
            feat_f0_flat = feat_f0_unfold.view(B, L * WW, C)
            feat_f1_flat = feat_f1_unfold.view(B, L * WW, C)
            
            # Verify window size
            W_f = int(WW ** 0.5)  # Should be 5
           
        elif len(feat_f0_unfold.shape) == 3:  # [B, L, C] - Flat feature format
            feat_f0_flat = feat_f0_unfold
            feat_f1_flat = feat_f1_unfold
            B, L, C = feat_f0_flat.shape
           
            
        else:
            raise ValueError(f"Unsupported feature dimension: {feat_f0_unfold.shape}")
        
        # Ensure feature dimension matches
        # Fine-matchingKeep the same spatial resolution as Coarse!
        hw_c = data.get('hw0_c', (8, 8))  # Coarse resolution
        if isinstance(hw_c, torch.Size):
            hw_c = (hw_c[0], hw_c[1])
        
        # Fine uses the same spatial resolution, but with a larger window
        H_fine = hw_c[0]  # Same as Coarse: 8
        W_fine = hw_c[1]  # Same as Coarse: 8
        
           
        # Feature projection and normalization
        feat_f0_proj = self.feat_proj(feat_f0_flat)
        feat_f1_proj = self.feat_proj(feat_f1_flat)
        
        # For window features, compute flow separately in each window
        if len(feat_f0_unfold.shape) == 4:
            # Keep window structure, do not aggregate!
            feat_f0_windowed = feat_f0_proj.view(B, L, WW, self.dim)
            feat_f1_windowed = feat_f1_proj.view(B, L, WW, self.dim)
            
            
            
            # Directly return window features for subsequent per-window processing
            return feat_f0_windowed, feat_f1_windowed, True  # True indicates window format
        else:
            # For flat features, reshape to image format
            feat_f0_aggregated = feat_f0_proj
            feat_f1_aggregated = feat_f1_proj
            
            # Reshape to image format
            feat_f0 = feat_f0_aggregated.view(B, H_fine, W_fine, self.dim).permute(0, 3, 1, 2)
            feat_f1 = feat_f1_aggregated.view(B, H_fine, W_fine, self.dim).permute(0, 3, 1, 2)
            
            # Update data dictionary
            data['hw0_f'] = (H_fine, W_fine)
            data['hw1_f'] = (H_fine, W_fine)
            
            return feat_f0, feat_f1, False  # False indicates not window format
    
    def _compute_windowed_flow(self, feat_f0_windowed, feat_f1_windowed, data):
        """Compute flow separately in each 5x5 window - vectorized version, processing all windows at once"""
        
        B, L, WW, C = feat_f0_windowed.shape
        W_w = int(WW ** 0.5)
        device = feat_f0_windowed.device
        
        # Determine spatial resolution
        hw_c = data.get('hw0_c', (8, 8))
        H_f, W_f = hw_c

        # 1. Get the original, complete coarse flow field [B, 2, H_f, W_f]
        coarse_flow = data['flow_c'].to(device)
        # [B, 2, H_f, W_f] -> [B, H_f, W_f, 2] -> [B, L, 2]
        coarse_flow_points = coarse_flow.permute(0, 2, 3, 1).reshape(B, L, 2)

        # Vectorization: concatenate all window features into a large batch
        # [B, L, WW, C] -> [B*L, WW, C]
        window_feat0_batch = feat_f0_windowed.view(B * L, WW, C)
        window_feat1_batch = feat_f1_windowed.view(B * L, WW, C)
        
        # Reshape to spatial window format [B*L, C, W_w, W_w]
        window_feat0_batch = window_feat0_batch.permute(0, 2, 1).view(B * L, C, W_w, W_w)
        window_feat1_batch = window_feat1_batch.permute(0, 2, 1).view(B * L, C, W_w, W_w)
        
        # Extract initial flow vectors for all windows [B*L, 2]
        flow_init_batch = coarse_flow_points.view(B * L, 2)
        # Expand to window size [B*L, 2, W_w, W_w]
        flow_init_batch = flow_init_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, W_w, W_w)

        # Process all windows at once
        center_flow_batch, center_confidence_batch, iter_center_flow_batch, iter_center_confidence_batch = checkpoint(self._compute_local_window_flow,window_feat0_batch, window_feat1_batch, flow_init_batch)
        
        
        flows = center_flow_batch.view(B, L, 2)
        confidences = center_confidence_batch.view(B, L, 1)
        
        # Process iteration results
        iter_flow_list = []
        iter_confidence_field = []
        for iter_idx in range(self.num_iterations):
            iter_flow_list.append(iter_center_flow_batch[iter_idx].view(B, L, 2))
            iter_confidence_field.append(iter_center_confidence_batch[iter_idx].view(B, L, 1))
        
        # Reshape to spatial format
        flow_field = flows.view(B, H_f, W_f, 2).permute(0, 3, 1, 2)  # [B, 2, H_f, W_f]
        confidence_field = confidences.view(B, H_f, W_f, 1).permute(0, 3, 1, 2)  # [B, 1, H_f, W_f]
        
        # Process iteration results
        iter_flow_f = []
        iter_conf_f = []
        for iter_idx in range(self.num_iterations):
            iter_flow_f.append(iter_flow_list[iter_idx].view(B, H_f, W_f, 2).permute(0, 3, 1, 2))
            iter_conf_f.append(iter_confidence_field[iter_idx].view(B, H_f, W_f, 1).permute(0, 3, 1, 2))
        
        # Update data dictionary
        data['hw0_f'] = (H_f, W_f)
        data['hw1_f'] = (H_f, W_f)
       
        return flow_field, confidence_field, iter_flow_f, iter_conf_f

    def _compute_local_window_flow(self, window_feat0_batch, window_feat1_batch, flow_init_batch):
        """5×5Refine flow -  batch """
        B_total, C, W_w, W_w = window_feat0_batch.shape  # B_total = B * L
        device = window_feat0_batch.device
        
        iter_center_flow_list = []
        iter_center_confidence_list = []
        
        flow = flow_init_batch
        
        for i in range(self.num_iterations):
            flow_before=flow.clone()
            flow = self.flow_refinement(window_feat0_batch, window_feat1_batch, flow)
            
            max_displacement = W_w // 2
            delta = flow - flow_before
            delta = torch.clamp(delta, -max_displacement, max_displacement)
            flow = flow_before + delta
            current_conf = self.confidence_estimator(window_feat0_batch, window_feat1_batch, flow)
            
            # Vectorization: find the maximum confidence position in each batch
            conf_flat = current_conf.view(B_total, -1)  # [B_total, W_w*W_w]
            max_conf_indices = torch.argmax(conf_flat, dim=1)
            max_y = max_conf_indices // W_w
            max_x = max_conf_indices % W_w
            
            
            current_iter_center_flow = flow[torch.arange(B_total), :, max_y, max_x]
            current_iter_center_confidence = current_conf[torch.arange(B_total), 0, max_y, max_x].unsqueeze(1)
            
            iter_center_flow_list.append(current_iter_center_flow)
            iter_center_confidence_list.append(current_iter_center_confidence)
        
        # Final confidence
        final_confidence = self.confidence_estimator(window_feat0_batch, window_feat1_batch, flow)
        final_confidence_flat = final_confidence.view(B_total, -1)
        max_conf_indices = torch.argmax(final_confidence_flat, dim=1)
        max_y = max_conf_indices // W_w
        max_x = max_conf_indices % W_w
        
        center_flow = flow[torch.arange(B_total), :, max_y, max_x]
        center_confidence = final_confidence[torch.arange(B_total), 0, max_y, max_x].unsqueeze(1)
        
        return center_flow, center_confidence, iter_center_flow_list, iter_center_confidence_list

    def _edge_aware_smoothing(self, flow, confidence):
        """Flow smoothing"""
        # Smoothed by confidence weighting
        weighted_flow = flow * confidence
        
        # Edge-aware smoothing
        smoothed_flow = self.edge_aware_smoother( weighted_flow)
        
        # Blend with original flow
        alpha = 0.7 # Smoothing weight
        final_flow = alpha * smoothed_flow + (1 - alpha) * flow
        
        return final_flow
    
    def _upsample_to_image_resolution(self, flow_field, data):
        """Upsample flow to full image resolution - Fine has same resolution as Coarse"""
        hw_f = data.get('hw0_f', flow_field.shape[-2:])  # Fine resolution (same as Coarse)
        
        # Original image resolution = Fine resolution x 8 (since Fine is also 1/8 resolution)
        hw_i = data.get('hw0_i', (hw_f[0] * 8, hw_f[1] * 8))
        
        if hw_i[0] > hw_f[0]:
            scale_factor = 8.0  # Fine to original image is 8x
            
            # Upsample flow field
            flow_full = F.interpolate(
                flow_field, 
                size=hw_i, 
                mode='bilinear', 
                align_corners=True
            )
            
            # Scale flow values
            flow_full = flow_full * scale_factor
            
            #print(f"[DEBUG] Flow upsampling: {hw_f} -> {hw_i}, Scaling factor: {scale_factor}")
        else:
            flow_full = flow_field
            #print(f"[DEBUG] No need to upsample, keep resolution: {hw_f}")
        
        return flow_full


class FeatureFlowAttention(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 **kwargs,
                 ):
        super(FeatureFlowAttention, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow,
                local_window_attn=False,
                local_window_radius=2,
                **kwargs,
                ):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow,
                                                  local_window_radius=local_window_radius)

        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # a note: the ``correct'' implementation should be:
        # ``query = self.q_proj(query), key = self.k_proj(query)''
        # this problem is observed while cleaning up the code
        # however, this doesn't affect the performance since the projection is a linear operation,
        # thus the two projection matrices for key can be merged
        # so I just leave it as is in order to not re-train all models :)
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return out

    def forward_local_window_attn(self, feature0, flow,
                                  local_window_radius=2,
                                  ):
        assert flow.size(1) == 2
        assert local_window_radius > 0

        b, c, h, w = feature0.size()

        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b * h * w, 1, c)  # [B*H*W, 1, C]

        kernel_size = 2 * local_window_radius + 1

        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)

        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size,
                                   padding=local_window_radius)  # [B, C*(2R+1)^2), H*W]

        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(
            0, 3, 4, 1, 2).reshape(b * h * w, c, kernel_size ** 2)  # [B*H*W, C, (2R+1)^2]

        flow_window = F.unfold(flow, kernel_size=kernel_size,
                               padding=local_window_radius)  # [B, 2*(2R+1)^2), H*W]

        flow_window = flow_window.view(b, 2, kernel_size ** 2, h, w).permute(
            0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2, 2)  # [B*H*W, (2R+1)^2, 2]

        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  # [B*H*W, 1, (2R+1)^2]

        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, flow_window).view(b, h, w, 2).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        return out