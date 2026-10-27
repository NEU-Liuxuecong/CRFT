"""
flow_bridge.py —— 路线二共用产流管线（缓存 / 训练 / 评测全部调这里的函数）

目的：保证"训练时缓存的流"与"测试时实时的流"逐环节一致（这是本路线最大的坑）。

注意（重要坑）：cv2.imread 读不了含中文的路径（本机仓库在 D:\\epm\\复现CRFT 下），
而训练时用的是相对路径（cwd=CRFT-main）。所以本模块 import 时即 os.chdir(CRFT_ROOT)，
所有"喂给 cv2 的路径"一律用相对路径 —— 与训练环境完全一致。
输出文件（缓存 .pth 等）用 Python open 写，中文路径无碍，放 refiner512 目录下。

只读使用 CRFT-main（sys.path 插入），不做任何修改。
"""
import os
import os.path as osp
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# --- Kechaba patch: resolve paths from configs/machine_paths.json (algorithm untouched) ---
from paths import machine_paths
_KROOT = Path(__file__).resolve().parents[2]
_CFG = machine_paths()
CRFT_ROOT = (_KROOT / _CFG['crft_root']).resolve()
HERE = Path(__file__).resolve().parent        # refiner512 目录（先算好，再 chdir）

os.chdir(CRFT_ROOT)                            # 复现训练时的 cwd，之后全用相对路径
sys.path.insert(0, str(CRFT_ROOT))

from src.config.default import _CN as cfg                       # noqa: E402
from src.lightning.lightning_crft import PL_CRFT                # noqa: E402
from src.datasets.RoadScence import RoadScenceDataset           # noqa: E402
from src.datasets.osdataset import OSDataset                    # noqa: E402
from src.utils.dataset import read_megadepth_gray, read_flow     # noqa: E402

# 两个 64 基线最优权重（各自数据集上从头训的）
CKPT_RS = Path(_CFG['crft_ckpt_roadscene'])   # Kechaba: from config
CKPT_OS = Path(_CFG['crft_ckpt_os'])          # Kechaba: from config
DATA_ROOT = {
    'roadscene': _CFG['data_root_roadscene'],
    'os': _CFG['data_root_os']
}
CACHE_DIR = HERE / 'cache'                      # 绝对路径（torch 读写安全）
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class OSDatasetFixed(OSDataset):
    """
    修复原版 OSDataset 的 bug（你的本地改动引入，上游硬编码版没有）：
    cands 只在 `if sub_dir is None:` 块内赋值 -> 第二次 __getitem__ 必
    UnboundLocalError（实测复现；crft_os_b4 是用上游版训的，没踩到）。
    这里在 __init__ 一次性确定 sub_dir，__getitem__ 逻辑与原版等价。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode_dir = osp.join(self.root_dir, self.mode)
        warped = osp.join(mode_dir, 'warped')
        if osp.isdir(warped):
            cands = [osp.join('warped', d) for d in os.listdir(warped)
                     if osp.isdir(osp.join(warped, d))]
        else:
            cands = [d for d in os.listdir(mode_dir) if osp.isdir(osp.join(mode_dir, d))]
        self.sub_dir = cands[0] if cands else ''

    def __getitem__(self, idx):
        pair_id = self.pair_list[idx]
        base = osp.join(self.root_dir, self.mode, self.sub_dir)
        opt_path = osp.join(base, 'image_pair', f'{pair_id}_opt.tif')
        sar_path = osp.join(base, 'image_pair', f'{pair_id}_sar_warped.tif')
        flow_path = osp.join(base, 'truth_flow', f'{pair_id}.flo')

        image0, mask0, scale0 = read_megadepth_gray(
            opt_path, self.img_resize, self.df, self.img_padding, augment_fn=None)
        image1, mask1, scale1 = read_megadepth_gray(
            sar_path, self.img_resize, self.df, self.img_padding, augment_fn=None)

        if osp.exists(flow_path):
            flow = read_flow(flow_path)
            if self.img_resize is not None:
                h, w = image0.shape[1:]
                H0, W0 = flow.shape[1:]
                scale_y, scale_x = h / H0, w / W0
                flow_resized = F.interpolate(flow[None], size=(h, w), mode='bilinear',
                                             align_corners=True)[0]
                flow_resized[0] = flow_resized[0] * scale_x
                flow_resized[1] = flow_resized[1] * scale_y
                flow = flow_resized
        else:
            flow = None

        data = {
            'image0': image0, 'image1': image1,
            'image0_path': opt_path, 'image1_path': sar_path,
            'flow': flow, 'scale0': scale0, 'scale1': scale1,
            'dataset_name': 'OSDataset', 'pair_id': idx,
            'pair_names': (f'{pair_id}_opt.tif', f'{pair_id}_sar_warped.tif'),
        }
        if mask0 is not None:
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(
                    torch.stack([mask0, mask1], dim=0)[None].float(),
                    scale_factor=self.coarse_scale, mode='nearest',
                    recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
        return data

# 你的 64 基线最优权重（RoadScene，val AEPE 1.067 @ epoch 486）
CACHE_DIR = HERE / 'cache'                      # 绝对路径（torch 读写安全）
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def set_cache_dir(tag):
    """Kechaba: resolve the flow cache from configs/machine_paths.json (tag: 'os' | 'roadscene')."""
    global CACHE_DIR
    entry = _CFG.get('cache_dirs', {}).get(tag or 'roadscene', {})
    if entry:
        CACHE_DIR = Path(next(iter(entry.values()))).parent
    else:
        CACHE_DIR = HERE / ('cache' if not tag else f'cache_{tag}')
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def build_config(dataset='roadscene'):
    """复刻 train.py 的 config merge —— 架构与匹配配置必须与训练时一致"""
    cfg.merge_from_file('configs/crft/outdoor/visible_thermal.py')
    if dataset == 'os':
        cfg.merge_from_file('configs/data/osdataset_640.py')
    else:
        cfg.merge_from_file('configs/data/roadscence_512.py')
    return cfg.clone()


def build_backbone(ckpt=None, dataset='roadscene', device='cuda'):
    """冻结的 64 基线主干（按数据集自动选 ckpt）。PL_CRFT 自己加载（自动剥 'matcher.' 前缀）"""
    if ckpt is None:
        ckpt = CKPT_OS if dataset == 'os' else CKPT_RS
    pl = PL_CRFT(build_config(dataset), pretrained_ckpt=str(ckpt))
    pl.eval().to(device)
    for p in pl.parameters():
        p.requires_grad_(False)
    return pl


def flow_to_512(flow_64, size=(512, 512)):
    """
    final_flow(64x64, 64像素单位) -> 512x512, 512像素单位
    空间插值 + 流值同比例放大（与 fine_matching.py:381 / lightning test_step:416-419 同一惯例：
    64 图上的 1 像素位移 = 512 图上的 8 像素位移，数值必须 ×8 才能和原生 .flo 同单位）
    """
    ratio = size[0] / flow_64.shape[-2]
    flow_512 = F.interpolate(flow_64, size=size, mode='bilinear', align_corners=False)
    return flow_512 * ratio


@torch.no_grad()
def backbone_flow(pl, batch, device='cuda'):
    """跑一遍冻结主干，返回 final_flow（输入图像分辨率，输入像素单位）"""
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    pl.matcher(batch)
    return batch['flow_f_full']


def make_dataset(split, img_resize, dataset='roadscene'):
    """
    split      : 'train' / 'val' / 'test'
    img_resize : 64  -> 主干训练域输入（与训练完全一致，缓存/评测喂主干用）
                 512 -> 原生 512（resize 无操作、flow 缩放恒等，喂头+GT 用）
    dataset    : 'roadscene'（3 通道可见光/红外）或 'os'（1 通道光学/SAR，用修复版数据集类）
    """
    root = DATA_ROOT[dataset]
    if dataset == 'os':
        return OSDatasetFixed(root_dir=root, list_path=f'{root}/{split}_list.txt',
                              mode=split, img_resize=img_resize, df=8, img_padding=True)
    return RoadScenceDataset(root_dir=root, list_path=f'{root}/{split}_list.txt',
                             mode=split, img_resize=img_resize, df=8, img_padding=True)


def aepe(flow_pred, flow_gt):
    """AEPE = 平均端点误差（与 lightning _compute_flow_based_metrics 同口径）"""
    d = flow_pred - flow_gt
    return torch.sqrt(d[0] ** 2 + d[1] ** 2).mean().item()
