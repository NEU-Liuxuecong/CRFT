"""RoadScene RS2 adaptation: Stage G (global) -> Stage L (local), initialised from a
*newly trained* OS D1 given by --init-os-d1.

This mirrors the archived RS2 recipe but never reads the author's OS weight:
  G: 2000 optimizer steps, local frozen, global_match_loss, AdamW lr 1e-4 (fixed, no scheduler)
  L: 2000 optimizer steps, global frozen, local trained on
       0.5*geometry_loss(local(initial_flow)) + 0.5*geometry_loss(local(global_coarse))
     coarse flow comes from the multi-peak robust estimate under no_grad (no GT)
  selection: every 500 steps on the FULL val23, metric = selected D1 AEPE BEFORE F13
"""
import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch

ORIG = Path.cwd()
PKG = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2] / "src").is_dir() \
    else Path(__file__).resolve().parents[1]
for p in (str(PKG / "src"), str(PKG / "src" / "runtime"), str(PKG / "src" / "f13")):
    sys.path.insert(0, p)
import plugin                                                    # noqa: E402
from paths import machine_paths                                   # noqa: E402
from geo_core_v1 import geometry_loss, dense_similarity           # noqa: E402
from global_rescue import global_match_loss                        # noqa: E402
from diverse_rescue import DiverseRescueRefiner, robust_pose_diverse  # noqa: E402

DATASET = "roadscene"
SEED = 66
STEPS = 2000
VAL_EVERY = 500


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def stats(v):
    v = sorted(v); n = len(v)
    med = v[n // 2] if n % 2 else .5 * (v[n // 2 - 1] + v[n // 2])
    return dict(n=n, mean=sum(v) / n, median=med, max=v[-1], lt1=sum(x < 1 for x in v),
                lt2=sum(x < 2 for x in v), ge5=sum(x >= 5 for x in v), ge10=sum(x >= 10 for x in v))


class Pairs(torch.utils.data.Dataset):
    def __init__(self, split):
        cfg = machine_paths()["cache_dirs"][DATASET]
        cpath = Path(cfg[split])
        with open(cpath, "rb") as f:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        if ck.get("unit") != "512 native px" or ck.get("split") != split:
            raise RuntimeError("cache unit/split mismatch")
        self.base = plugin.make_datasets(DATASET, split, img_resize=512)
        names = [self.base[i]["pair_names"][0] for i in range(len(self.base))]
        norm = lambda x: x[:-len("_visible.tif")] if x.endswith("_visible.tif") else x
        idx = {norm(q): i for i, q in enumerate(ck["pairs"])}
        missing = [w for w in map(norm, names) if w not in idx]
        if missing:
            raise RuntimeError(f"cache missing {len(missing)} pairs")
        if ck.get("ckpt_sha256") and ck["ckpt_sha256"] != sha(machine_paths()["crft_ckpt_roadscene"]):
            raise RuntimeError("cache/CRFT mismatch")
        self.initials = [ck["initial"][idx[norm(n)]].float() for n in names]
        self.gts = [ck["gt"][idx[norm(n)]].float() for n in names]
        self.names, self.cache_path = names, cpath

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        d = self.base[i]
        return dict(image0=d["image0"], image1=d["image1"], flow_init=self.initials[i],
                    flow=self.gts[i], name=self.names[i])


def geoms(it, dev):
    im0, im1 = it["image0"][None], it["image1"][None]
    g = lambda t: t[0] if t.shape[1] == 1 else (0.299 * t[0][0] + 0.587 * t[0][1] + 0.114 * t[0][2])[None]
    return g(im0).unsqueeze(0).float().to(dev), g(im1).unsqueeze(0).float().to(dev)


@torch.no_grad()
def val_selected(d1, pairs, dev, limit=0):
    d1.eval()
    n = len(pairs) if not limit else min(limit, len(pairs))
    vals, sel = [], []
    for i in range(n):
        it = pairs[i]
        g0, g1 = geoms(it, dev)
        out, aux = d1(g0, g1, it["flow_init"][None].to(dev), return_aux=True)
        vals.append(float((out - it["flow"][None].to(dev)).norm(dim=1).mean()))
        sel.append(int(bool(aux["selection"][0]["selected_rescue"])))
    return vals, sel


def main():
    ap = argparse.ArgumentParser(description="RS2 adaptation from a newly trained OS D1")
    ap.add_argument("--init-os-d1", type=Path, required=True, help="newly trained OS D1 (assembled)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--stage", choices=["g", "l", "both"], default="both")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--val-every", type=int, default=VAL_EVERY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="tiny sample/step budget, marked SMOKE")
    a = ap.parse_args()
    assert a.device == "cuda" and torch.cuda.is_available()
    dev = a.device
    out = (a.outdir if a.outdir.is_absolute() else ORIG / a.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    init = a.init_os_d1 if a.init_os_d1.is_absolute() else ORIG / a.init_os_d1
    if not init.is_file():
        raise SystemExit(f"--init-os-d1 not found: {init}")
    with open(init, "rb") as f:
        payload = torch.load(f, map_location="cpu", weights_only=False)
    if payload.get("protocol", {}).get("dataset") != "os":
        raise SystemExit("init-os-d1 must be an OS checkpoint")
    init_sha = sha(init)

    logf = open(out / "rs2_train_log.jsonl", "a", encoding="utf-8")
    def log(m):
        print(m, flush=True); logf.write(str(m) + "\n"); logf.flush()

    train, val = Pairs("train"), Pairs("val")
    if a.smoke:
        train = torch.utils.data.Subset(train, list(range(8)))
        val = torch.utils.data.Subset(val, list(range(3)))
    set_seed(SEED)
    d1 = DiverseRescueRefiner(**payload["architecture"]).to(dev)
    d1.load_state_dict(payload["state_dict"], strict=True)
    log(json.dumps(dict(event="start", init_os_d1=str(init), init_os_d1_sha256=init_sha,
                        steps=a.steps, val_every=a.val_every, smoke=bool(a.smoke),
                        n_train=len(train), n_val=len(val), device=dev)))
    for p in d1.parameters():
        p.requires_grad_(False)

    def run_stage(kind, steps):
        if kind == "g":
            target = list(d1.global_matcher.parameters())
            frozen = list(d1.local.parameters())
            d1.global_matcher.train(); d1.local.eval()
        else:
            target = list(d1.local.parameters())
            frozen = list(d1.global_matcher.parameters())
            d1.local.train(); d1.global_matcher.eval()
        for p in target:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(target, lr=1e-4, weight_decay=1e-4)   # fixed lr, no scheduler
        rng = random.Random(SEED + (0 if kind == "g" else 1000))
        frozen_before = {n: p.detach().clone() for n, p in
                         (d1.local.named_parameters() if kind == "g" else d1.global_matcher.named_parameters())}
        best = (float("inf"), 0, None)
        t0, run, nstep = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        order, oi = [], 0
        for step in range(1, steps + 1):
            if oi % max(len(train), 1) == 0:
                order = list(range(len(train))); rng.shuffle(order); oi = 0
            it = train[order[oi]]; oi += 1
            g0, g1 = geoms(it, dev)
            init_flow = it["flow_init"][None].to(dev)
            gt = it["flow"][None].to(dev)
            if kind == "g":
                rec = d1.global_matcher(g0, g1)
                loss, _ = global_match_loss(rec, gt)
            else:
                with torch.no_grad():
                    _, aux0 = d1.local(g0, g1, init_flow, return_aux=True)
                    rec = d1.global_matcher(g0, g1)
                    theta, _ = robust_pose_diverse(rec, aux0["theta"].detach(), g0.shape[-2],
                                                   g0.shape[-1], d1.hypotheses)
                    coarse = dense_similarity(theta, g0.shape[-2], g0.shape[-1])
                _, aux_a = d1.local(g0, g1, init_flow, return_aux=True)
                la, _ = geometry_loss(aux_a, gt, 0.1)
                _, aux_b = d1.local(g0, g1, coarse, return_aux=True)
                lb, _ = geometry_loss(aux_b, gt, 0.1)
                loss = 0.5 * la + 0.5 * lb
            (loss / 4.0).backward()
            run += float(loss.detach()); nstep += 1
            if step % 4 == 0 or step == steps:
                k = 4 if step % 4 == 0 else step % 4
                if k != 4:
                    for p in target:
                        if p.grad is not None:
                            p.grad.mul_(4.0 / k)
                torch.nn.utils.clip_grad_norm_(target, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            if step % a.val_every == 0 or step == steps or a.smoke:
                vals, sel = val_selected(d1, val, dev, 3 if a.smoke else 0)
                m = sum(vals) / len(vals)
                improved = m < best[0] - 1e-12
                if improved:
                    best = (m, step, {k: v.detach().clone() for k, v in d1.state_dict().items()})
                frozen_ok = all(torch.equal(frozen_before[n], p.detach()) for n, p in
                                (d1.local.named_parameters() if kind == "g"
                                 else d1.global_matcher.named_parameters()))
                log(json.dumps(dict(stage=kind, step=step, train_loss=run / max(nstep, 1),
                                    val_selected_aepe=m, best=best[0], best_step=best[1],
                                    improved=improved, frozen_unchanged=frozen_ok,
                                    rescue_used=sum(sel), sec=round(time.time() - t0, 1),
                                    smoke=bool(a.smoke))))
                run, nstep, t0 = 0.0, 0, time.time()
            if a.smoke and step >= 8:
                break
        for p in target:
            p.requires_grad_(False)
        return best

    if a.stage in ("g", "both"):
        best = run_stage("g", a.steps)
        if best[2] is not None:
            d1.load_state_dict(best[2], strict=True)
        with open(out / "rs2_g_best.pt", "wb") as f:
            torch.save(dict(step=best[1], val=best[0], state_dict=d1.state_dict(),
                            rng=rng_state(), init_os_d1_sha256=init_sha), f)
        log(json.dumps(dict(event="g_done", best_step=best[1], best_val=best[0],
                            ckpt_sha256=sha(out / "rs2_g_best.pt"))))
    if a.stage in ("l", "both"):
        best = run_stage("l", a.steps)
        if best[2] is not None:
            d1.load_state_dict(best[2], strict=True)
        with open(out / "rs2_l_best.pt", "wb") as f:
            torch.save(dict(step=best[1], val=best[0], state_dict=d1.state_dict(),
                            rng=rng_state(), init_os_d1_sha256=init_sha), f)
        payload_out = dict(kind="frozen_d1_v1", architecture=payload["architecture"],
                           state_dict=d1.state_dict(),
                           protocol=dict(dataset="roadscene", from_scratch=False,
                                         init_os_d1_sha256=init_sha,
                                         stages=f"G{best[1]}+L{best[1]} (best node)",
                                         note="adapted from THIS run's OS D1"))
        with open(out / "rs2_d1.ckpt", "wb") as f:
            torch.save(payload_out, f)
        log(json.dumps(dict(event="l_done", best_step=best[1], best_val=best[0],
                            rs2_d1_sha256=sha(out / "rs2_d1.ckpt"),
                            init_os_d1_sha256=init_sha)))
    logf.close()


if __name__ == "__main__":
    main()
