import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR, CosineAnnealingWarmRestarts


def build_optimizer(model, config):
    name = config.TRAINER.OPTIMIZER
    lr = config.TRAINER.TRUE_LR

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=config.TRAINER.SGD_DECAY)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, weight_decay=config.TRAINER.RMSPROP_DECAY)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.TRAINER.SCHEDULER

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config.TRAINER.MSLR_MILESTONES, gamma=config.TRAINER.MSLR_GAMMA)})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config.TRAINER.COSA_TMAX)})
    elif name == 'CosineAnnealingWarmRestarts':
        scheduler.update(
            {'scheduler': CosineAnnealingWarmRestarts(optimizer, 
                                                    T_0=config.TRAINER.COSWR_T0 if hasattr(config.TRAINER, 'COSWR_T0') else 50,
                                                    T_mult=config.TRAINER.COSWR_T_MULT if hasattr(config.TRAINER, 'COSWR_T_MULT') else 2,
                                                    eta_min=config.TRAINER.COSWR_ETA_MIN if hasattr(config.TRAINER, 'COSWR_ETA_MIN') else 1e-6)})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config.TRAINER.ELR_GAMMA)})
    else:
        raise NotImplementedError()

    return scheduler
