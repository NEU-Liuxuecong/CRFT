
import math
import argparse
from pathlib import Path
from loguru import logger
import time 

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.config.default import get_cfg_defaults
from src.utils.misc import setup_gpus
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_crft import PL_CRFT

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os


def main():
    # Simple argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='crft-train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--pretrained_ckpt', type=str,default='.ckpt')
    args = parser.parse_args() 
    
    print(f"Starting Training")
    print(f"   - Experiment: {args.exp_name}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - GPUs: {args.gpus}")
    print(f"   - Pretrained checkpoint: {args.pretrained_ckpt}")

    # Load config
    config = get_cfg_defaults() #val 0r not
    config.merge_from_file('configs/crft/outdoor/visible_thermal.py')#
    config.merge_from_file('configs/data/osdataset_640.py')#
    #config.merge_from_file('configs/data/roadscence_512.py')
    # Override configs
    config.TRAINER.USE_WANDB = False  # Disable wandb
    config.TRAINER.MAX_EPOCHS = args.epochs
    
    pl.seed_everything(config.TRAINER.SEED)

    # Setup GPU
    if torch.cuda.is_available() and args.gpus > 0:
        device = 'gpu'
        accelerator = 'gpu'
        devices = args.gpus
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        accelerator = 'cpu'
        devices = 1
        print("Using CPU (GPU not available)")
    
    # Scale learning rate 
    config.TRAINER.WORLD_SIZE = devices
    config.TRAINER.TRUE_BATCH_SIZE = devices * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    
    print(f"raining configuration:")
    print(f"   - True batch size: {config.TRAINER.TRUE_BATCH_SIZE}")
    print(f"   - Learning rate: {config.TRAINER.TRUE_LR:.6f}")
    print(f"   - Warmup steps: {config.TRAINER.WARMUP_STEP}")

    
    # Create model
    try:
        model = PL_CRFT(config, pretrained_ckpt=args.pretrained_ckpt)
        print("Model loaded with pretrained weights")
    except Exception as e:
        print(f"Failed to load pretrained weights: {e}")
        model = PL_CRFT(config)
        print("Model created without pretrained weights")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data module
    data_module = MultiSceneDataModule(args, config)
    print("Data module created")
    
    # Setup logging and callbacks
    logger_tb = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name)
    ckpt_dir = Path(logger_tb.log_dir) / 'checkpoints'
    
    callbacks = [ 
        
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            monitor='val_AEPE',
            save_top_k=15,
            mode='min',
            save_last=True,
            dirpath=str(ckpt_dir),
            filename='{epoch:02d}-{val_AEPE:.3f}'
        )
    ]
    
    # Create trainer (Lightning 1.3.5 compatible)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=devices if device == 'gpu' else 0,  
        logger=logger_tb,  
        callbacks=callbacks,
        gradient_clip_val=1,
        precision=32, 
        amp_backend='native',  # Use 3+2-bit precision to avoid overflow 
        check_val_every_n_epoch=1,
        log_every_n_steps=1000
    )
    
    print("Trainer created")
    print(f"Logs will be saved to: {logger_tb.log_dir}")
    
    # Start training
    try:
        print("Starting training...")
        
        start_time = time.time()  
        
        trainer.fit(model, datamodule=data_module)
        
        end_time = time.time() 
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        

        avg_epoch_time = total_time / args.epochs
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        
        print("Training completed successfully!")
        print(f"Best model saved at: {ckpt_dir}")
        
        # Print final metrics
        if trainer.callback_metrics:
            print("\nFinal metrics:")
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    print(f"   - {key}: {value.item():.4f}")
                else:
                    print(f"   - {key}: {value}")
                    
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
