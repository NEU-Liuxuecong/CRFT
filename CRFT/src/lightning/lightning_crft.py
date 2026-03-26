from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from src.utils.dataset import read_image_rgb
from src.lightning import data
plt.switch_backend('agg')

import flow_viz
from src.crft import CRFT
from src.losses.crft_loss import CRFTLoss#,RobustFlowLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from imshow import save_flow_visualizations, save_flowviz_bundle
from imshow_1 import save_polygon_figure


class PL_CRFT(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.crft_cfg = lower_config(_config['crft'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: CRFT
        self.matcher = CRFT(config=_config['crft']) 
        self.loss = CRFTLoss(_config) 

        # Pretrained weights
        if pretrained_ckpt: 
            state_dict = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)['state_dict']
            self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir

    def configure_ddp(self):
        self.trainer.model._set_static_graph()
        return self.trainer.model

        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config) 
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step( 
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):  
        
        with self.profiler.profile("CRFT"):
            self.matcher(batch)
            
        with self.profiler.profile("Compute losses"):
            self.loss(batch)
    
    def _compute_metrics(self, batch):
        """Compute metrics based on optical flow."""
        with self.profiler.profile("Compute metrics"):
            metrics = {
                'mae': [],
                'rmse': [],
                'sr': [],
                'AEPE':[]
            }
            
            # Get SR threshold from config or use default
            sr_threshold = getattr(self.config.METRICS, 'SR_THRESHOLD', 3.0)
            
            if 'flow_f_full' in batch and 'flow' in batch:
                flow_pred = batch['flow_f_full']  # [B, 2, H, W]
                flow_gt = batch['flow']           # [B, 2, H, W]
                
                for b in range(batch['image0'].size(0)):
                    flow_pred_b = flow_pred[b]  # [2, H, W]
                    flow_gt_b = flow_gt[b]      # [2, H, W]
                    
                    if flow_pred_b.shape != flow_gt_b.shape:
                        flow_pred_b = F.interpolate(
                            flow_pred_b.unsqueeze(0), 
                            size=flow_gt_b.shape[-2:], 
                            mode='bilinear', 
                            align_corners=True
                        ).squeeze(0)
                    
                    mae, rmse, sr ,AEPE= self._compute_flow_based_metrics(
                        flow_pred_b, flow_gt_b, sr_threshold
                    )
                    
                    metrics['mae'].append(mae)
                    metrics['rmse'].append(rmse)
                    metrics['sr'].append(sr)
                    metrics['AEPE'].append(AEPE)
            else:
                for b in range(batch['image0'].size(0)):
                    metrics['mae'].append(float('inf'))
                    metrics['rmse'].append(float('inf'))
                    metrics['sr'].append(0.0)
                    metrics['AEPE'].append(float('inf'))
            
            ret_dict = {'metrics': metrics}
            return ret_dict
    
    def _compute_flow_based_metrics(self, flow_pred, flow_gt, sr_threshold):
        """
        
        Args:
            flow_pred: [2, H, W]
            flow_gt: [2, H, W]
        """
        flow_diff = flow_pred - flow_gt
        
        euclidean_distance = torch.sqrt(flow_diff[0]**2 + flow_diff[1]**2)
        sr_mask = euclidean_distance < sr_threshold  # [H, W]
        sr = sr_mask.float().mean().item() * 100.0
        
        if sr_mask.sum() > 0:
            l1_distance = torch.abs(flow_diff[0]) + torch.abs(flow_diff[1])  # [H, W]
            mae = l1_distance.mean().item()
            AEPE= (torch.sqrt(torch.abs(flow_diff[0])**2+torch.abs(flow_diff[1])**2)).mean().item()

            squared_distance = flow_diff[0]**2 + flow_diff[1]**2  # [H, W]
            rmse = torch.sqrt(squared_distance[sr_mask].mean()).item()
            
        else:
            mae = float('inf')
            rmse = float('inf')
            AEPE=float('inf')
        #print('mae ',mae)
        return mae, rmse, sr, AEPE

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        ret_dict_train = self._compute_metrics(batch)
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:# no distributed
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.log(f'train{k}',v,on_step=True,on_epoch=False,prog_bar=True)
                if self.config.TRAINER.USE_WANDB:
                    self.log(f'wandb_train{k}',v,on_step=True,on_epoch=False)

            # figures
            '''if self.config.TRAINER.ENABLE_PLOTTING: 
                #compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, v in figures.items():
                    # Get TensorBoard logger
                    if isinstance(self.logger, list):
                        tensorboard_logger = self.logger[0]
                    else:
                        tensorboard_logger = self.logger
                    tensorboard_logger.experiment.add_figure(f'train_match/{k}',v,self.global_step)'''
        return {'loss': batch['loss'],
                'loss_scalars': batch['loss_scalars'],
                ** ret_dict_train}

    def training_epoch_end(self, outputs):
        # Step 1: Aggregate loss scalars from all GPUs
        all_loss_scalars = defaultdict(list)
        for output in outputs:
            if 'loss_scalars' in output:
                for k, v in output['loss_scalars'].items():
                    all_loss_scalars[k].append(v)
        
        _metrics = [o['metrics'] for o in outputs if 'metrics' in o]
        if _metrics:
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
        else:
            metrics = {}

        mae_values = [m for m in metrics.get('mae', []) if not np.isinf(m)]
        rmse_values = [m for m in metrics.get('rmse', []) if not np.isinf(m)]
        AEPE_values = [m for m in metrics.get('AEPE', []) if not np.isinf(m)]
        sr_values = metrics.get('sr', [])
        train_metrics = {
            'mae': np.mean(mae_values) if mae_values else float('inf'),
            'rmse': np.mean(rmse_values) if rmse_values else float('inf'),
            'AEPE': np.mean(AEPE_values) if AEPE_values else float('inf'),
            'sr': np.mean(sr_values) if sr_values else 0.0
        }
        self.log('trainmae', torch.tensor(train_metrics['mae'], device=self.device), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('trainAEPE', torch.tensor(train_metrics['AEPE'], device=self.device), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('trainrmse', torch.tensor(train_metrics['rmse'], device=self.device), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('trainsr', torch.tensor(train_metrics['sr'], device=self.device), on_epoch=True, prog_bar=True, sync_dist=True)

        if self.trainer.global_rank == 0:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            
            if isinstance(self.logger, list):
                tensorboard_logger = self.logger[0]
            else:
                tensorboard_logger = self.logger

            tensorboard_logger.experiment.add_scalar('train/avg_loss_on_epoch', avg_loss, global_step=self.current_epoch)
            
            if all_loss_scalars:
                for k, v in all_loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    tensorboard_logger.experiment.add_scalar(f'train/{k}', mean_v, global_step=self.current_epoch)
                    print(f'train/{k}', mean_v) 

            tensorboard_logger.experiment.add_scalar('train/mae', train_metrics['mae'], global_step=self.current_epoch)
            tensorboard_logger.experiment.add_scalar('train/AEPE', train_metrics['AEPE'], global_step=self.current_epoch)
            tensorboard_logger.experiment.add_scalar('train/sr', train_metrics['sr'], global_step=self.current_epoch)

            if self.config.TRAINER.USE_WANDB and isinstance(self.logger, list) and len(self.logger) > 1:
                wandb_logger = self.logger[1]
                wandb_logger.log_metrics({'train/avg_loss_on_epoch': avg_loss}, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        # no loss calculation for VisTir during val
        if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
            with self.profiler.profile("CRFT"):
                self.matcher(batch)
        else:
            self._trainval_inference(batch)
        
        ret_dict = self._compute_metrics(batch)
        
        
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        '''figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0 and batch_idx %100==0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE, ret_dict=ret_dict)
            if self.trainer.global_rank == 0 and figures[self.config.TRAINER.PLOT_MODE]:
                save_dir = f"validation_figures/epoch_{self.current_epoch}"
                os.makedirs(save_dir, exist_ok=True)
                
                for i, fig in enumerate(figures[self.config.TRAINER.PLOT_MODE]):
                    save_path = os.path.join(save_dir, f"batch_{batch_idx}_sample_{i}.png")
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)  '''
        if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
            return {
                **ret_dict,
                #'figures': figures,
            }
        else:
            return {
                **ret_dict,
                'loss_scalars': batch['loss_scalars'],
                #'figures': figures,
            }
        
    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1
            
            # Aggregate metrics flatten metrics list
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            _loss_scalars = [o['loss_scalars'] for o in outputs if 'loss_scalars' in o]
            if _loss_scalars:
                loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}
            
            # Compute mean metrics
            mae_values=[m for m in metrics['mae']if not np.isinf(m)] # Filter out infinite values, keep only meaningful MAE values
            AEPE_values=[m for m in metrics['AEPE']if not np.isinf(m)] 
            rmse_values=[m for m in metrics['rmse']if not np.isinf(m)] 
            sr_values=[m for m in metrics['sr']if not np.isinf(m)]
            #print('sr_value',sr_values)
            val_metrics = {
                'mae': np.mean(mae_values)if mae_values else float('inf'),
                'rmse': np.mean(rmse_values) if rmse_values else float('inf'),
                'AEPE': np.mean(AEPE_values) if AEPE_values else float('inf'),
                'sr': np.mean(sr_values)
            }
            #print('val_metrics',val_metrics['sr'])
            # Log metrics
            if self.trainer.global_rank == 0:
                # Log using PyTorch Lightning's self.log
                # Get TensorBoard logger
                if isinstance(self.logger, list):
                    tensorboard_logger = self.logger[0]
                else:
                    tensorboard_logger = self.logger
                
                # Log validation metrics using the same method as train/avg_loss_on_epoch
                tensorboard_logger.experiment.add_scalar(
                    'val/mae', val_metrics['mae'],
                    global_step=self.current_epoch)
                
                tensorboard_logger.experiment.add_scalar(
                    'val/AEPE', val_metrics['AEPE'],
                    global_step=self.current_epoch)
                tensorboard_logger.experiment.add_scalar(
                    'val/sr', val_metrics['sr'],
                    global_step=self.current_epoch)
                
                 # Log validation loss for overfitting monitoring
                if _loss_scalars:
                    for k, v in loss_scalars.items():
                        mean_v = torch.stack(v).mean()
                        tensorboard_logger.experiment.add_scalar(
                            f'val/{k}', mean_v,
                            global_step=self.current_epoch)
                        
            # Also log using self.log for ModelCheckpoint compatibility
            self.log('val_mae', torch.tensor(val_metrics['mae'], device=self.device), on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_rmse', torch.tensor(val_metrics['rmse'], device=self.device), on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_AEPE', torch.tensor(val_metrics['AEPE'], device=self.device), on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_sr', torch.tensor(val_metrics['sr'], device=self.device), on_epoch=True, prog_bar=True, sync_dist=True)
            
            # Log validation loss for ModelCheckpoint
            if _loss_scalars:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.log(f'val{k}', mean_v.to(self.device), on_epoch=True, prog_bar=True, sync_dist=True)

                # Log to WandB if enabled
                if self.config.TRAINER.USE_WANDB and isinstance(self.logger, list) and len(self.logger) > 1:
                    wandb_logger = self.logger[1]
                    metrics_dict = {
                        'valmae': val_metrics['mae'],
                        'valrmse': val_metrics['rmse'],
                        'valAEPE': val_metrics['AEPE'],
                        'valsr': val_metrics['sr']}
                     # Add validation loss to wandb
                    if _loss_scalars:
                        for k, v in loss_scalars.items():
                            mean_v = torch.stack(v).mean()
                            metrics_dict[f'val{k}'] = mean_v
                    wandb_logger.log_metrics(metrics_dict, cur_epoch)
                
                logger.info('\n' + pprint.pformat(val_metrics))
            multi_val_metrics.update(val_metrics)
        
        return multi_val_metrics

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("CRFT"):
            self.matcher(batch)
        ret_dict = self._compute_metrics(batch)
        '''
        # Visualize optical flow
        if batch_idx % 1 == 0:  # Save visualization results every 1 batch
            for i in range(batch['image0'].size(0)):  # Iterate through each sample in the batch
                # Get low-resolution optical flow and high-resolution images
                image1 = batch['image0'][i]  # Optical image (low resolution)
                image2 = batch['image1'][i]  # SAR image (low resolution)
                flow_pred = batch['flow_f_full'][i]  # Predicted optical flow (low resolution)
                flow_gt = batch['flow'][i]  # Ground truth optical flow (low resolution)

                # Check image dimensions
                if image1.dim() == 5:
                    image1 = image1.squeeze(0)
                if image2.dim() == 5:
                    image2 = image2.squeeze(0)

                # Get original high-resolution image paths
                image0_path = batch['image0_path'][i]
                image1_path = batch['image1_path'][i]
                img0_orig_tensor, _, _ = read_image_rgb(image0_path)
                img1_orig_tensor, _, _ = read_image_rgb(image1_path)

                # High-resolution dimensions
                orig_h, orig_w = img0_orig_tensor.shape[1], img0_orig_tensor.shape[2]

                # Interpolate optical flow from low resolution to high resolution
                flow_pred_high_res = F.interpolate(
                    flow_pred.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False
                ).squeeze(0)
                flow_gt_high_res = F.interpolate(
                    flow_gt.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False
                ).squeeze(0)

                # Scale optical flow magnitude
                scale_h = orig_h / flow_pred.shape[1]
                scale_w = orig_w / flow_pred.shape[2]
                flow_pred_high_res[0, :, :] *= scale_w
                flow_pred_high_res[1, :, :] *= scale_h
                flow_gt_high_res[0, :, :] *= scale_w
                flow_gt_high_res[1, :, :] *= scale_h

                # Convert to visualization format
                image1_high_res = img0_orig_tensor.unsqueeze(0).to(self.device)
                image2_high_res = img1_orig_tensor.unsqueeze(0).to(self.device)

                # Save directory
                save_dir = os.path.join(self.dump_dir, f"batch_{batch_idx}_sample_{i}")
                os.makedirs(save_dir, exist_ok=True)
                
                # Call visualization functions in imshow.py
                save_flow_visualizations(
                    image1_high_res, image2_high_res, flow_pred_high_res, flow_gt_high_res, save_dir, sample_idx=i
                )
                save_flowviz_bundle(
                    image1_high_res, image2_high_res, flow_pred_high_res, flow_gt_high_res, save_dir, sample_idx=i
                )
                save_polygon_figure(
                    image1_high_res, image2_high_res, flow_gt_high_res, flow_pred_high_res, i, save_dir
                )
                
                data_for_plotting = {
                        'image0': img0_orig_tensor.unsqueeze(0).to(self.device),
                        'image1': img1_orig_tensor.unsqueeze(0).to(self.device),
                        'flow_f_full': batch['flow_f_full'][i].unsqueeze(0) if 'flow_f_full' in batch else None,
                        'flow': batch['flow'][i].unsqueeze(0) if 'flow' in batch else None,
                        'pair_names': (batch['pair_names'][0][i], batch['pair_names'][1][i]),
                    }
                data_for_plotting = {k: v for k, v in data_for_plotting.items() if v is not None}
                single_ret_dict = {'metrics': {k: [v[i]] for k, v in ret_dict['metrics'].items()}}
                
                figures = make_matching_figures(data_for_plotting, self.config, mode='evaluation', ret_dict=single_ret_dict)
                save_dir = os.path.join(self.dump_dir, f"batch_{batch_idx}_sample_{i}")
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(data_for_plotting['pair_names'][0]))[0]
                for fig_idx, fig in enumerate(figures[self.config.TRAINER.PLOT_MODE]):
                    save_path = os.path.join(save_dir, f"evaluation.png")
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)'''
        return {**ret_dict}
        

    def test_epoch_end(self, outputs):
        # Aggregate metrics
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
        # Filter out infinite values, keep only meaningful MAE values
        mae_values=[m for m in metrics['mae']if not np.isinf(m)] 
        rmse_values=[m for m in metrics['rmse']if not np.isinf(m)] 
        sr_values=[m for m in metrics['sr']if not np.isinf(m)]
        aepe_values=[m for m in metrics['AEPE']if not np.isinf(m)]
        
        # Use NumPy to simplify threshold statistics
        if aepe_values:
            aepe_np = np.array(aepe_values)
            thresholds = [5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            
            # Print in original format (two per line approximately)
            for i in range(0, len(thresholds), 2):
                t1 = thresholds[i]
                acc1 = (aepe_np < t1).mean()
                
                if i + 1 < len(thresholds):
                    t2 = thresholds[i+1]
                    acc2 = (aepe_np < t2).mean()
                    print(f'{t1}px: {acc1:.4f} {t2}px: {acc2:.4f}')
                else:
                    print(f'{t1}px: {acc1:.4f}')

        # Compute mean metrics
        test_metrics= {
                'mae': np.mean(mae_values)if mae_values else float('inf'),
                'rmse': np.mean(rmse_values) if rmse_values else float('inf'),
                'AEPE': np.mean(aepe_values) if aepe_values else float('inf'),
                'sr': np.mean(sr_values) if sr_values else 0.0
            } 
        print(f"MAE: {np.mean(test_metrics['mae']):.4f} ")
        print(f"AEPE: {np.mean(test_metrics['AEPE']):.4f} ")
        print(f"RMSE: {np.mean(test_metrics['rmse']):.2f} ")
        print(f"Success Rate: {np.mean(test_metrics['sr']):.2f}% ")
        
        if self.trainer.global_rank == 0:
            logger.info('\n' + pprint.pformat(test_metrics))
