from configs.data.base import cfg

# OSDataset configuration
cfg.DATASET.TRAIN_DATA_SOURCE = "OSDataset"
cfg.DATASET.VAL_DATA_SOURCE = "OSDataset"
cfg.DATASET.TEST_DATA_SOURCE = "OSDataset"

# training data config
cfg.DATASET.TRAIN_DATA_ROOT = "/Dataset/OSdataset"
cfg.DATASET.TRAIN_LIST_PATH = "/Dataset/OSdataset/train_list.txt"
#contain the number of the image pairs
# validation data config  
cfg.DATASET.VAL_DATA_ROOT = "/Dataset/OSdataset"
cfg.DATASET.VAL_LIST_PATH = "/Dataset/OSdataset/val_list.txt"

# testing data config
cfg.DATASET.TEST_DATA_ROOT = "/Dataset/OSdataset"
cfg.DATASET.TEST_LIST_PATH = "/Dataset/OSdataset/test_list.txt"

# OSDataset specific options
cfg.DATASET.OS_IMG_RESIZE =64  # resize the longer edge to this size
cfg.DATASET.OS_IMG_PAD = True    # pad to square
cfg.DATASET.OS_DF = 8           # image size division factor

# dataset config
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0  # no overlap filtering for aligned images

# metrics config for OSDataset
cfg.METRICS.SR_THRESHOLD = 10000.0  # threshold for success rate in pixels
cfg.METRICS.USE_FULL_IMAGE = True  # evaluate on full image
cfg.METRICS.FLOW_SCALE = 1.0  # scale flow values by 0.1 to match expected translation