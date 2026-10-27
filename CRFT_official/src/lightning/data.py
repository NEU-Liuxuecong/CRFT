from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    DistributedSampler
)

from src.utils.augment import build_augmentor
from src.datasets.osdataset import OSDataset
from src.datasets.RoadScence import RoadScenceDataset


class MultiSceneDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.train_data_source = config.DATASET.TRAIN_DATA_SOURCE
        self.val_data_source = config.DATASET.VAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating 
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_pose_root = config.DATASET.TRAIN_POSE_ROOT  # (optional)
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.train_intrinsic_path = config.DATASET.TRAIN_INTRINSIC_PATH
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_pose_root = config.DATASET.VAL_POSE_ROOT  # (optional)
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
        # testing 
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_pose_root = config.DATASET.TEST_POSE_ROOT  # (optional)
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH

        # 2. dataset config
        # general options
        self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score
        self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']
        self.coarse_scale = 1 / config.CRFT.RESOLUTION[0]  # 0.125. for training crft.

        # OSDataset options
        self.os_img_resize = getattr(config.DATASET, 'OS_IMG_RESIZE',64) 
        self.os_img_pad = getattr(config.DATASET, 'OS_IMG_PAD', True)
        self.os_df = getattr(config.DATASET, 'OS_DF', 8)
        #RoadScenceDataset 
        self.rs_img_resize = getattr(config.DATASET, 'RS_IMG_RESIZE', 64)
        self.rs_img_pad = getattr(config.DATASET, 'RS_IMG_PAD', True)
        self.rs_df = getattr(config.DATASET, 'RS_DF', 8)

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True) 
        }
        self.val_loader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.test_loader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }
        
        # 4. sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT
        
        # (optional) RandomSampler for debugging

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            if dist.is_initialized():
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
                logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
            else:
                self.world_size = 1
                self.rank = 0
                logger.info("Distributed training not initialized, using single process")
        except Exception as e:
                self.world_size = 1
                self.rank = 0
                logger.warning(f"Failed to get distributed info: {e} (set world_size=1 and rank=0)")

        if stage == 'fit':#val ot train or test
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                self.train_npz_root,
                self.train_list_path,
                self.train_intrinsic_path,
                mode='train',
                min_overlap_score=self.min_overlap_score_train,
                pose_dir=self.train_pose_root)
            # setup multiple (optional) validation subsets
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [self.val_npz_root for _ in range(len(self.val_list_path))]
                for npz_list, npz_root in zip(self.val_list_path, self.val_npz_root):
                    self.val_dataset.append(self._setup_dataset(
                        self.val_data_root,
                        npz_root,
                        npz_list,
                        self.val_intrinsic_path,
                        mode='val',
                        min_overlap_score=self.min_overlap_score_test,
                        pose_dir=self.val_pose_root))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_npz_root,
                    self.val_list_path,
                    self.val_intrinsic_path,
                    mode='val',
                    min_overlap_score=self.min_overlap_score_test,
                    pose_dir=self.val_pose_root)
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_npz_root,
                self.test_list_path,
                self.test_intrinsic_path,
                mode='test',
                min_overlap_score=self.min_overlap_score_test,
                pose_dir=self.test_pose_root)
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self,
                       data_root,
                       split_npz_root,
                       scene_list_path,
                       intri_path,
                       mode='train',
                       min_overlap_score=0.,
                       pose_dir=None):
        """ Setup train / val / test set"""
        # Check if this is OSDataset
        data_source = getattr(self, f'{mode}_data_source', self.train_data_source)
        if data_source == 'OSDataset':
            # For OSDataset, directly create the dataset without npz processing
            return OSDataset(
                root_dir=data_root,
                list_path=scene_list_path,
                mode=mode,
                img_resize=self.os_img_resize,
                df=self.os_df,
                img_padding=self.os_img_pad,
                coarse_scale=self.coarse_scale
            )
        elif data_source == 'RoadScenceDataset':
            return RoadScenceDataset(
                root_dir=data_root,
                list_path=scene_list_path,
                mode=mode,
                img_resize=self.rs_img_resize,
                df=self.rs_df,
                img_padding=self.rs_img_pad,
                coarse_scale=self.coarse_scale
            )
        
        raise ValueError(f"Unknown dataset: {data_source}")

    def train_dataloader(self):
        """ Build training dataloader for OSDataset / RoadScenceDataset. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        
        train_data_source = getattr(self, 'train_data_source', None)
        if train_data_source == 'OSDataset' or train_data_source == 'RoadScenceDataset':
            sampler = None
            if self.world_size > 1:
                from torch.utils.data import DistributedSampler
                sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            raise ValueError(f"Unsupported train_data_source: {train_data_source}")
        
        dataloader = DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
        return dataloader
    
    def val_dataloader(self):
        """ Build validation dataloader. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        if not isinstance(self.val_dataset, abc.Sequence):
            if self.world_size > 1:
                sampler = DistributedSampler(self.val_dataset, shuffle=False)
            else:
                sampler = None
            return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)
        else:
            dataloaders = []
            for dataset in self.val_dataset:
                if self.world_size > 1:
                    sampler = DistributedSampler(dataset, shuffle=False)
                else:
                    sampler = None
                dataloaders.append(DataLoader(dataset, sampler=sampler, **self.val_loader_params))
            return dataloaders

    def test_dataloader(self, *args, **kwargs):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        if self.world_size > 1:
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            sampler = None
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)
