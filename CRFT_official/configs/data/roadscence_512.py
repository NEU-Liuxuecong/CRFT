from src.config.default import _CN as cfg



cfg.DATASET.TRAIN_DATA_SOURCE = 'RoadScenceDataset'
cfg.DATASET.VAL_DATA_SOURCE = 'RoadScenceDataset'
cfg.DATASET.TEST_DATA_SOURCE = 'RoadScenceDataset'
'''
cfg.DATASET.TRAIN_DATA_ROOT = '/Dataset/MixBigRS'
cfg.DATASET.TRAIN_LIST_PATH = '/Dataset/MixBigRS/train_list.txt'
cfg.DATASET.VAL_DATA_ROOT = '/Dataset/MixBigRS'
cfg.DATASET.VAL_LIST_PATH = '/Dataset/MixBigRS/val_list.txt'
cfg.DATASET.TEST_DATA_ROOT = '/Dataset/MixBigRS'
cfg.DATASET.TEST_LIST_PATH = '/Dataset/MixBigRS/test_list.txt'
'''

# cfg.DATASET.TRAIN_DATA_ROOT = '/Dataset/MixBigRS'
# cfg.DATASET.TRAIN_LIST_PATH = '/Dataset/MixBigRS/train_list.txt'
# cfg.DATASET.VAL_DATA_ROOT = '/Dataset/MixBigRS'
# cfg.DATASET.VAL_LIST_PATH = '/Dataset/MixBigRS/val_list.txt'
# cfg.DATASET.TEST_DATA_ROOT = '/Dataset/MixBigRS'
# cfg.DATASET.TEST_LIST_PATH = '/Dataset/MixBigRS/test_list.txt'
# OSDataset specific options
cfg.DATASET.RS_IMG_RESIZE =64  # resize the longer edge to this size
cfg.DATASET.RS_IMG_PAD = True    # pad to square
cfg.DATASET.RS_DF = 8           # image size division factor

# dataset config
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0  # no overlap filtering for aligned images

# metrics config for OSDataset
cfg.METRICS.SR_THRESHOLD = 10000.0  # threshold for success rate in pixels
cfg.METRICS.USE_FULL_IMAGE = True  # evaluate on full image
cfg.METRICS.FLOW_SCALE = 1.0  # scale flow values by 0.1 to match expected translation