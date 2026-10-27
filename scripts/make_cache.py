"""Generate CRFT initial-flow caches from raw images - no dependency on author caches.

  python make_cache.py --dataset os        --split train --out cache/os_train.pth
  python make_cache.py --dataset roadscene --split val   --out cache/rs_val.pt

The bridge is applied exactly once: bilinear to 512 (align_corners=False) and the flow is
multiplied by 8 (64-pixel units -> 512 native pixels).

Output formats (what the trainers read):
  os        : dict{pair_name -> float32 (2,512,512)}
  roadscene : dict{split, pairs, initial(N,2,512,512), gt, unit, shape,
                   bridge_sha256, ckpt_sha256, crft_ckpt, generated_by}
"""
import argparse, hashlib, json, sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
import plugin                                          # noqa: E402
from paths import machine_paths                        # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["os", "roadscene"], required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = full split (only for smoke you use >0)")
    a = ap.parse_args()
    dev = a.device
    backbone = plugin.build_backbone(a.dataset, dev)
    small = plugin.make_datasets(a.dataset, a.split, img_resize=64)
    native = plugin.make_datasets(a.dataset, a.split, img_resize=512)
    n = len(native) if not a.limit else min(a.limit, len(native))
    from torch.utils.data import default_collate
    flows, gts, names = [], [], []
    with torch.no_grad():
        for i in range(n):
            s, d = small[i], native[i]
            if tuple(s["pair_names"]) != tuple(d["pair_names"]):
                raise RuntimeError("small/native sample mismatch at index " + str(i))
            batch = default_collate([{k: v for k, v in s.items() if k != "flow"}])
            flows.append(plugin.bridge.flow_to_512(
                plugin.bridge.backbone_flow(backbone, batch, device=dev))[0].cpu())
            gts.append(d["flow"].cpu())
            names.append(d["pair_names"][0])
            if (i + 1) % 64 == 0:
                print(f"  {i+1}/{n}", flush=True)
    bridge = plugin.bridge
    prov = dict(bridge_sha256=sha(Path(bridge.__file__).resolve()),
                ckpt_sha256=sha(Path(bridge.CKPT_OS if a.dataset == "os" else bridge.CKPT_RS)),
                crft_ckpt=str(bridge.CKPT_OS if a.dataset == "os" else bridge.CKPT_RS),
                unit="512 native px", shape=[2, 512, 512],
                generated_by="make_cache.py")
    out = a.out if a.out.is_absolute() else Path.cwd() / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.dataset == "os":
        payload = {nm: f for nm, f in zip(names, flows)}
    else:
        payload = dict(split=a.split, pairs=names, initial=torch.stack(flows),
                       gt=torch.stack(gts), **prov)
    with open(out, "wb") as f:
        torch.save(payload, f)
    meta = dict(dataset=a.dataset, split=a.split, n=n, out=str(out), sha256=sha(out), **prov)
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
