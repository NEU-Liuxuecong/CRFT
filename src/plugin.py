"""Official 512 plug-in pipeline: frozen CRFT -> original bridge -> D1 -> optional F13.

Frozen assets (paths from configs/machine_paths.json):
  OS          : weights/os/frozen_d1.ckpt                  (D1, 555,359 params)
  RoadScene   : weights/roadscene/best_d1_roadscene_full.pth (RS2 full-branch D1)
  F13         : src/f13/geometry_prior512.py (zero-parameter, same for both)

Design notes (see docs/网络架构与代码对应.md):
  * D1's top-level forward is @torch.no_grad by design; training uses the branch paths.
  * disable_rescue=True skips only the global rescue + selection, NOT the whole D1.
  * F13.disable=True returns the D1 flow unchanged.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

from paths import BRIDGEDIR, F13DIR, KROOT, RUNTIME, machine_paths, ensure_sys_path

EXPECTED = dict(
    d1_os="41ad43fafbdf197fe70615a6c967ec27f8847e84fa879b0f199fc98933859990",
    d1_roadscene="f077d10d8c73539b8b5f3751f39c42e765b7ed6da2bd3fc28b28b9aeb26cc6d6",
    f13="fde6083b4a7165573ffa73481a62c1c739744de06d30a0b96f125a3b0ff49d5c",
)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ensure_sys_path()
if str(BRIDGEDIR) not in sys.path:
    sys.path.insert(0, str(BRIDGEDIR))

import flow_bridge as bridge                      # noqa: E402  (patched: path constants only)
from geometry_prior512 import GeometryPrior512     # noqa: E402
from diverse_rescue import DiverseRescueRefiner    # noqa: E402


def load_payload(path):
    with open(path, "rb") as f:                    # Python file object: safe for non-ASCII paths
        return torch.load(f, map_location="cpu")


def load_d1(dataset, device="cuda", verify=True, weight_path=None):
    """Build the D1 structure and load weights with strict=True.

    weight_path=None -> the official frozen weight (hash verified).
    weight_path given -> a user checkpoint (e.g. a new run); the hash is recorded, not enforced.
    """
    cfg = machine_paths()
    path = Path(weight_path) if weight_path else Path(cfg["d1_os"] if dataset == "os" else cfg["d1_roadscene"])
    if verify and weight_path is None:
        want = EXPECTED["d1_os"] if dataset == "os" else EXPECTED["d1_roadscene"]
        got = sha256(path)
        if got != want:
            raise RuntimeError(f"D1 weight hash mismatch for {dataset}: {got}")
    payload = load_payload(path)
    arch = dict(payload["architecture"])
    model = DiverseRescueRefiner(**arch)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict load failed: missing={missing} unexpected={unexpected}")
    model = model.to(device).eval().requires_grad_(False)
    info = dict(dataset=dataset, weight=str(path), weight_sha256=sha256(path),
                official=(weight_path is None),
                step=payload.get("step"), architecture=arch,
                local_config=payload.get("local_config") or model.local.config,
                params_total=sum(p.numel() for p in model.parameters()),
                params_local=sum(p.numel() for p in model.local.parameters()),
                params_global=sum(p.numel() for p in model.global_matcher.parameters()))
    return model, info


def load_prior(device="cuda", verify=True):
    cfg = machine_paths()
    f13_path = F13DIR / "geometry_prior512.py"
    if verify and sha256(f13_path) != EXPECTED["f13"]:
        raise RuntimeError("F13 source hash mismatch")
    prior = GeometryPrior512().to(device).eval()
    info = dict(path=str(f13_path), sha256=sha256(f13_path),
                params=sum(p.numel() for p in prior.parameters()))
    return prior, info


def build_backbone(dataset, device="cuda"):
    """Frozen CRFT backbone (64 input) selected by dataset."""
    cfg = machine_paths()
    ckpt = cfg["crft_ckpt_os"] if dataset == "os" else cfg["crft_ckpt_roadscene"]
    pl = bridge.PL_CRFT(bridge.build_config(dataset), pretrained_ckpt=str(ckpt))
    pl.eval().to(device)
    for p in pl.parameters():
        p.requires_grad_(False)
    return pl


def make_datasets(dataset, split, img_resize=512):
    return bridge.make_dataset(split, img_resize=img_resize, dataset=dataset)


def to_gray_for_d1(img, channels):
    """D1 expects 1 channel. OS loader already yields 1; RoadScene is RGB -> fixed luminance.

    No extra rescaling: each dataset keeps its own loader range (OS [0,1], RoadScene [0,255]).
    """
    if channels == 1:
        return img
    return (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)


@torch.no_grad()
def run_pipeline(backbone, d1, prior, small_item, native_item, device="cuda",
                 use_f13=True, disable_rescue=False, return_aux=False):
    """One pair through the official chain. GT never enters; it is only returned separately."""
    from torch.utils.data import default_collate
    batch = default_collate([{k: v for k, v in small_item.items() if k != "flow"}])
    initial = bridge.flow_to_512(bridge.backbone_flow(backbone, batch, device=device))
    img0 = native_item["image0"][None].to(device).float()
    img1 = native_item["image1"][None].to(device).float()
    ch = img0.shape[1]
    g0 = to_gray_for_d1(img0[0], ch).unsqueeze(0)
    g1 = to_gray_for_d1(img1[0], ch).unsqueeze(0)
    d1_out = d1(g0, g1, initial.float(), return_aux=True, disable_rescue=disable_rescue)
    d1_flow, aux = d1_out if isinstance(d1_out, tuple) else (d1_out, {})
    f13_flow = prior(d1_flow) if use_f13 else d1_flow
    out = dict(initial=initial, d1=d1_flow, final=f13_flow, aux=aux,
               gt=native_item["flow"][None].to(device).float(),
               pair=native_item["pair_names"][0])
    return out


@torch.no_grad()
def run_bridge_only(backbone, prior, small_item, native_item, device="cuda", use_f13=True):
    """No plug-in at all: the bridge initial flow, optionally projected by F13."""
    from torch.utils.data import default_collate
    batch = default_collate([{k: v for k, v in small_item.items() if k != "flow"}])
    initial = bridge.flow_to_512(bridge.backbone_flow(backbone, batch, device=device))
    f13_flow = prior(initial) if use_f13 else initial
    return dict(initial=initial, d1=initial, final=f13_flow,
                gt=native_item["flow"][None].to(device).float(),
                pair=native_item["pair_names"][0], aux={})
