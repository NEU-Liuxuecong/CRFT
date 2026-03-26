import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import read_image_rgb, read_flow

class RoadScenceDataset(Dataset):
    def __init__(self,
                 root_dir,
                 list_path,
                 mode='train',
                 img_resize=64,
                 df=8,
                 img_padding=True,
                 **kwargs):
        """
        Manage RoadScence dataset with visible and infrared image pairs.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        
        self.coarse_scale = kwargs.get('coarse_scale', 0.125)
        
        with open(list_path, 'r') as f:
            self.pair_list = [line.strip() for line in f.readlines()]
        
        logger.info(f"RoadScenceDataset {mode} mode: {len(self.pair_list)} scenes loaded. Images will be loaded as RGB.")

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        pair_id = self.pair_list[idx]
        
        # Construct paths for RoadScence dataset
    
        "BigMixRoad"
        visible_path = osp.join(self.root_dir, self.mode, 'wrapped', 'image_pair', f'{pair_id}_rgb.tif')
        infrared_path = osp.join(self.root_dir, self.mode, 'wrapped', 'image_pair', f'{pair_id}_ir_warped.tif')
        flow_path = osp.join(self.root_dir, self.mode, 'wrapped', 'truth_flow', f'{pair_id}.flo')
        

        # Read images as 3-channel RGB tensors
        image0, mask0, scale0 = read_image_rgb(
            visible_path, self.img_resize, self.df, self.img_padding, augment_fn=None)
        image1, mask1, scale1 = read_image_rgb(
            infrared_path, self.img_resize, self.df, self.img_padding, augment_fn=None)
        
        # Read flow
        if osp.exists(flow_path):
            flow = read_flow(flow_path)
            if self.img_resize is not None:
                h, w = image0.shape[1:]
                H0, W0 = flow.shape[1:]
                scale_y, scale_x = h / H0, w / W0
                flow_resized = F.interpolate(flow[None], size=(h, w), mode='bilinear', align_corners=True)[0]
                flow_resized[0] *= scale_x
                flow_resized[1] *= scale_y
                flow = flow_resized
        else:
            h, w = image0.shape[1:]
            flow = torch.zeros((2, h, w), dtype=torch.float)

        data = {
            'image0': image0,
            'image1': image1,
            'image0_path': visible_path,
            'image1_path': infrared_path,
            'flow': flow,
            'scale0': scale0,
            'scale1': scale1,
            'dataset_name': 'RoadScence',
            'pair_id': idx,
            'pair_names': (f'{pair_id}_visible.tif', f'{pair_id}_infrared_warped.tif'),
        }
        if flow is not None:
            data['flow'] = flow
            
        if mask0 is not None:
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data