"""Live CRFT + existing D1 branch training; unchanged full D1/F13 inference.

FP32, batch size 1. Fine loss reaches CRFT through local D1. The discrete
global rescue selector is inference-only; its descriptors use existing CE.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src'), str(ROOT / 'src/runtime')]
from unified_os import UnifiedCRFTD1F13 as OSModel
from unified_roadscene import UnifiedCRFTD1F13 as RSModel
import plugin
from geo_core_v1 import geometry_loss
from global_rescue import global_match_loss
from train_independent import geoms, rng_state, rng_restore, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class JointModel(nn.Module):
    def __init__(self, dataset, checkpoint, device):
        super().__init__()
        self.inference = (OSModel if dataset == 'os' else RSModel)(checkpoint, device)
        self.device = device
        self.requires_grad_(True)
        self.train()

    @property
    def crft(self):
        return self.inference.crft

    @property
    def d1(self):
        return self.inference.d1

    def train(self, mode=True):
        super().train(mode)
        # Batch size 1 fine-tuning: retain pretrained running statistics, but
        # keep BN affine parameters trainable. Positional MLPs stay in train
        # mode so cached position tensors are recomputed every forward.
        if mode:
            for m in self.crft.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward_train(self, small, native):
        if small['pair_names'] != native['pair_names']:
            raise RuntimeError('64/512 pair mismatch')
        batch = default_collate([{k: v for k, v in small.items() if k != 'flow'}])
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if batch['image0'].shape[-2:] != (64, 64):
            raise ValueError('CRFT requires 64x64 input')
        self.crft.matcher(batch)  # deliberately bypass inference no_grad wrapper
        flow64 = batch['flow_f_full']
        if flow64.shape != (1, 2, 64, 64) or not flow64.requires_grad:
            raise RuntimeError('CRFT flow has wrong shape or no graph')
        initial = plugin.bridge.flow_to_512(flow64)  # exactly one x8
        g0, g1 = geoms(native, self.device)  # original luminance/range
        refined, aux = self.d1.local(g0, g1, initial, return_aux=True, detach_initial=False)
        record = self.d1.global_matcher(g0, g1)
        return dict(initial=initial, refined=refined, local_aux=aux, global_record=record,
                    flow64=flow64, gray_range=[float(g0.min()), float(g0.max())])


def gradient_stats(module):
    rows = {}
    for name, p in module.named_parameters():
        g = p.grad
        rows[name] = dict(has_grad=g is not None,
                          finite=g is None or bool(torch.isfinite(g).all()),
                          norm=0. if g is None else float(g.double().norm()),
                          nonzero=g is not None and bool(g.abs().sum() > 0))
    norms = [r['norm'] for r in rows.values()]
    return dict(norm=math.sqrt(sum(n*n for n in norms)),
                finite=all(r['finite'] for r in rows.values()),
                with_grad=sum(r['has_grad'] for r in rows.values()),
                without_grad=sum(not r['has_grad'] for r in rows.values()),
                nonzero=sum(r['nonzero'] for r in rows.values()), parameters=rows)


def snapshot(module):
    return {n: p.detach().cpu().clone() for n, p in module.named_parameters()}


def update_stats(module, before):
    rows = {n: float((p.detach().cpu().double()-before[n].double()).norm())
            for n, p in module.named_parameters()}
    return dict(norm=math.sqrt(sum(x*x for x in rows.values())),
                changed=sum(x > 0 for x in rows.values()), parameters=rows)


def paired_datasets(dataset, split):
    small = plugin.make_datasets(dataset, split, 64)
    native = plugin.make_datasets(dataset, split, 512)
    prefix = 'os' if dataset == 'os' else 'rs'
    delivery = (ROOT / f'data/lists/{prefix}_{split}.txt').read_text().splitlines()
    if small.pair_list != delivery or native.pair_list != delivery:
        raise RuntimeError('Dataset split differs from delivered list')
    return small, native


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=['os', 'roadscene'], required=True)
    ap.add_argument('--smoke', action='store_true', help='default 3 updates then save/reload/full inference')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seed', type=int, default=66)
    ap.add_argument('--lr-crft', type=float, default=1e-5)
    ap.add_argument('--lr-d1', type=float, default=1e-4)
    ap.add_argument('--lambda-coarse', type=float, default=.1)
    ap.add_argument('--lambda-global', type=float, default=.1)
    ap.add_argument('--outdir', type=Path)
    ap.add_argument('--resume', type=Path)
    args = ap.parse_args()
    steps = args.steps if args.steps is not None else (3 if args.smoke else 100)
    if steps < 1 or args.lambda_coarse < 0 or args.lambda_global <= 0:
        ap.error('steps >= 1, lambda-coarse >= 0, lambda-global > 0 required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed)
    out = (args.outdir or ROOT / f'runs/joint_{args.dataset}').resolve()
    if (ROOT / 'runs').resolve() not in out.parents:
        raise RuntimeError('Outputs must be inside experiment runs/')
    # No accidental checkpoint/log overwrite, including repeat smoke runs.
    out.mkdir(parents=True, exist_ok=False)
    logf = (out / 'train_log.jsonl').open('x')

    def log(row):
        line = json.dumps(row, allow_nan=False)
        print(line, flush=True)
        logf.write(line+'\n'); logf.flush()

    init = ROOT / ('weights/os/CRFT_OSdataset.ckpt' if args.dataset == 'os'
                   else 'weights/roadscene/CRFT_RoadScene.ckpt')
    checkpoint = args.resume.resolve() if args.resume else init
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['dataset'] != args.dataset:
        raise RuntimeError('Cross-dataset initialization refused')
    small, native = paired_datasets(args.dataset, 'train')
    val_small, val_native = paired_datasets(args.dataset, 'val')
    # IDs restart in each split; equal basenames do not imply equal files.
    train_root = (Path(native.root_dir)/native.mode).resolve()
    val_root = (Path(val_native.root_dir)/val_native.mode).resolve()
    if train_root == val_root:
        raise RuntimeError('Train/validation directories overlap')
    model = JointModel(args.dataset, checkpoint, 'cuda')

    if not args.resume:
        # Scheme B: keep pretrained CRFT, initialize a completely fresh D1.
        # Reconstruct the entire D1 so local and global branches are both fresh.
        d1_class = type(model.d1)
        model.inference.d1 = d1_class(
            **payload['d1_architecture']
        ).to('cuda')

        model.requires_grad_(True)
        model.train()

        print(
            '[Scheme B] Pretrained CRFT loaded; '
            'D1 local/global randomly initialized.',
            flush=True
        )

    opt = torch.optim.AdamW([dict(params=model.crft.parameters(), lr=args.lr_crft),
                             dict(params=model.d1.parameters(), lr=args.lr_d1)], weight_decay=1e-4)
    start = 0
    if args.resume:
        opt.load_state_dict(payload['optimizer_state_dict'])
        rng_restore(payload['rng_state'])
        start = payload['global_step']
        if payload['training_config']['seed'] != args.seed:
            raise RuntimeError('Resume seed mismatch')
    log(dict(event='start', dataset=args.dataset, init=str(checkpoint), init_sha256=sha(checkpoint),
             seed=args.seed, train_count=len(native), val_count=len(val_native),
             gpu=torch.cuda.get_device_name(), torch=torch.__version__, precision='fp32',
             gpu_total_mib=torch.cuda.get_device_properties(0).total_memory/2**20,
             config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             loss=dict(local_geometry_weight=1.,local_match_weight=.1,
                       coarse_weight=args.lambda_coarse,global_weight=args.lambda_global),
             normalization='dense mean over all 512x512 pixels; finite GT required',
             bn='eval running statistics; trainable affine parameters',
             f13='unchanged inference-only', global_rescue='discrete selector inference-only; existing CE training'))
    torch.cuda.reset_peak_memory_stats()
    last = None
    for step in range(start, start+steps):
        t0 = time.time()
        # A deterministic epoch permutation allows exact resume without a sampler cache.
        epoch, offset = divmod(step, len(native))
        order = list(range(len(native))); random.Random(args.seed+epoch).shuffle(order)
        index = order[offset]
        s, n = small[index], native[index]
        if any(not bool(n[k].all()) for k in ('mask0','mask1') if k in n):
            raise RuntimeError('Padded native images require an explicit dense loss mask policy')
        gt = n['flow'][None].to('cuda').float()
        if gt.shape != (1,2,512,512) or not bool(torch.isfinite(gt).all()) or bool((gt.abs() >= 1e8).any()):
            raise RuntimeError('Invalid/sentinel GT: explicit mask policy required before training')
        model.train(); opt.zero_grad(set_to_none=True)
        before_c, before_d = snapshot(model.crft), snapshot(model.d1)
        pred = model.forward_train(s, n)
        fine, fs = geometry_loss(pred['local_aux'], gt, match_weight=.1)
        coarse = torch.sqrt((pred['initial']-gt).square().sum(1)+1e-6).mean()
        glob, gs = global_match_loss(pred['global_record'], gt)
        loss = fine + args.lambda_coarse*coarse + args.lambda_global*glob
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite loss')
        if step == start:
            # Stronger than total-loss backward: prove D1 fine loss alone reaches
            # CRFT's actual backbone/coarse/fine matching parameters.
            fine.backward(retain_graph=True)
            proof = {name:gradient_stats(mod) for name,mod in {
                'crft':model.crft, 'backbone':model.crft.matcher.backbone,
                'coarse_matching':model.crft.matcher.coarse_matching,
                'fine_matching':model.crft.matcher.fine_matching,
                'd1_local':model.d1.local}.items()}
            (out/'fine_only_gradient_proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False))
            for name, stat in proof.items():
                if not stat['finite'] or stat['norm'] <= 0:
                    raise RuntimeError('Fine-only gradient failed: '+name)
            log(dict(event='fine_only_gradient_proof', **{k:{x:v[x] for x in ('norm','finite','with_grad','without_grad','nonzero')} for k,v in proof.items()}))
            opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = {name:gradient_stats(mod) for name,mod in {
            'crft':model.crft, 'd1':model.d1, 'crft_backbone':model.crft.matcher.backbone,
            'crft_transformer':model.crft.matcher.loftr_coarse,
            'crft_coarse':model.crft.matcher.coarse_matching,
            'crft_fine_process':model.crft.matcher.fine_process,
            'crft_fine_matching':model.crft.matcher.fine_matching,
            'd1_local':model.d1.local, 'd1_global':model.d1.global_matcher}.items()}
        for name,g in grads.items():
            if not g['finite'] or g['norm'] <= 0:
                raise RuntimeError('Missing/nonfinite parameter gradient: '+name)
        clip = float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        opt.step()
        updates = dict(crft=update_stats(model.crft,before_c),d1=update_stats(model.d1,before_d))
        if any(v['norm'] <= 0 for v in updates.values()):
            raise RuntimeError('Zero parameter update')
        detail = dict(gradients=grads, updates=updates)
        (out/f'gradients_step_{step+1}.json').write_text(json.dumps(detail,indent=2,allow_nan=False))
        last = dict(event='step',dataset=args.dataset,global_step=step+1,epoch=epoch,
                    pair=n['pair_names'][0],index=index,loss=float(loss.detach()),
                    fine=float(fine.detach()),coarse=float(coarse.detach()),global_match=float(glob.detach()),
                    geometry={k:float(v) for k,v in fs.items()},
                    gradients={k:{x:v[x] for x in ('norm','finite','with_grad','without_grad','nonzero')} for k,v in grads.items()},
                    updates={k:{x:v[x] for x in ('norm','changed')} for k,v in updates.items()},
                    clip_before=clip,lr=[g['lr'] for g in opt.param_groups],gray_range=pred['gray_range'],
                    gt_range=[float(gt.min()),float(gt.max())],gt_valid_pixels=512*512,
                    flow64_shape=list(pred['flow64'].shape),flow512_shape=list(pred['initial'].shape),
                    peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.time()-t0)
        log(last)
        del pred, fine, coarse, glob, loss, before_c, before_d, gt
    # Preserve exact unified inference schema and add complete resumable state.
    payload.update(kind='crft_d1_joint_v1', crft_state_dict=model.crft.state_dict(),
                   d1_state_dict=model.d1.state_dict(),optimizer_state_dict=opt.state_dict(),
                   scheduler_state_dict=None,global_step=start+steps,
                   epoch=(start+steps-1)//len(native),rng_state=rng_state(),
                   training_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                   joint_provenance=dict(source=str(checkpoint),sha256=sha(checkpoint),
                                         loss_position='local D1 before discrete rescue/F13',
                                         global_supervision='existing global_match_loss'))
    saved = out/('smoke.ckpt' if args.smoke else 'last.ckpt')
    torch.save(payload,saved)
    # Remove training graphs/cached positions before testing a newly constructed model.
    del model,opt,payload
    torch.cuda.empty_cache()
    restored = (OSModel if args.dataset=='os' else RSModel)(saved,'cuda')
    with torch.inference_mode():
        vs,vn=val_small[0],val_native[0]
        result=restored(vs,vn)
        if result.shape != (1,2,512,512) or not bool(torch.isfinite(result).all()):
            raise RuntimeError('Reloaded full-pipeline inference failed')
        val_aepe=float((result-vn['flow'][None].cuda()).norm(dim=1).mean())
    log(dict(event='saved_reloaded_inference',dataset=args.dataset,checkpoint=str(saved),
             sha256=sha(saved),reload_strict=True,full_d1_rescue_f13=True,
             split='val',pair=vn['pair_names'][0],aepe=val_aepe,
             note='One validation sample only; no accuracy improvement claim',
             peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))
    logf.close()


if __name__ == '__main__':
    main()
