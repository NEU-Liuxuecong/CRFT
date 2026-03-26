from src.config.default import _CN as cfg

cfg.CRFT.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.TRAINER.CANONICAL_LR = 3e-4
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24, 30, 36, 42]

# pose estimation - disabled for aligned images
cfg.TRAINER.RANSAC_PIXEL_THR = 100.0  # Very large threshold
cfg.TRAINER.EPI_ERR_THR = 100.0  # Very large threshold for aligned images

cfg.TRAINER.OPTIMIZER = "adamw"
#cfg.TRAINER.SGD_DECAY = 0.0001
#cfg.TRAINER.ADAM_DECAY = 0.005
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.CRFT.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3

cfg.TRAINER.USE_WANDB = False # use weight and biases

cfg.CRFT.MATCH_COARSE.BORDER_RM = 0

