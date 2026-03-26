import argparse
import pprint
from pathlib import Path
from loguru import logger as loguru_logger
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_crft import PL_CRFT
from src.utils.plotting import make_matching_figures
import flow_viz
loguru_logger = get_rank_zero_only_logger(loguru_logger)
import os

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ckpt_path', type=str, default='CRFT_OSdataset.ckpt')
    parser.add_argument(
        '--dump_dir', type=str, default='results',help='output directory for results')
    parser.add_argument(
        '--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--batch_size', type=int, default=16, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=16)
    parser.add_argument(
        '--thr', type=float, default=None, help='modify the coarse-level matching threshold.')
    args = parser.parse_args()
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file('configs/crft/outdoor/visible_thermal.py')
    config.merge_from_file('configs/data/osdataset_640.py')
    #config.merge_from_file('configs/data/roadscence_512.py')
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # tune when testing
    if args.thr is not None:
        config.CRFT.MATCH_COARSE.THR = args.thr

    loguru_logger.info(f"Args and config initialized!")

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_CRFT(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir)
    loguru_logger.info(f"CRFT-lightning ini-tialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    # lightning trainer
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, replace_sampler_ddp=False, logger=False)
    
    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module, verbose=False)

