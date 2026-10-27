"""Plug-in trained INDEPENDENTLY from random init for ONE dataset (no other dataset's weights).

Protocol (mirrors the original OS staged recipe, per instruction):
  L0  random GeometryRefiner, train local, <=120 ep, lr1e-3, mw .1, patience 30
  G1  GlobalRescueRefiner(local = THIS run's L0), train global_matcher only, <=60 ep,
      lr1e-4, patience 15, min_ep 30, original global_match_loss
  L1  GeometryRefiner from the SAME L0, train local, <=40 ep, lr1e-4, mw .5, patience 10,
      min_ep 15, original perturb_initial
  assemble  DiversityRescueRefiner(local = L1) + G1 global_matcher (strict)

Stage selection follows the original OS rule and never uses F13:
  L0/L1 -> local flow full-val AEPE;  G1 -> GlobalRescueRefiner full inference full-val AEPE.
The continuous F13 is used ONLY for the final method output / reporting.

Every torch.load is recorded and checked: allowed sources are this run's own checkpoints,
the RoadScene flow cache and the frozen RoadScene CRFT weight. Author plug-in weights
(weights/os, weights/roadscene, old RS2/A-B runs) are refused.
"""
import argparse, contextlib, copy, hashlib, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch

ORIG_CWD = Path.cwd()
def _find_root(start):
    for q in [start, *start.parents]:
        if (q / "configs" / "machine_paths.json").is_file():
            return q
    raise SystemExit("package root not found (configs/machine_paths.json)")


PKG = _find_root(Path(__file__).resolve())          # Kechaba/
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "src" / "runtime"))
sys.path.insert(0, str(PKG / "src" / "f13"))
import plugin                                        # noqa: E402
from paths import machine_paths                       # noqa: E402
from geo_core_v1 import GeometryRefiner, geometry_loss            # noqa: E402
from global_rescue import GlobalRescueRefiner, global_match_loss, perturb_initial  # noqa: E402
from diverse_rescue import DiverseRescueRefiner                   # noqa: E402
from geometry_prior512 import GeometryPrior512                     # noqa: E402

DATASET = "roadscene"   # overridden by --dataset before any use
ARCH = dict(global_grid=64, hypotheses=1024, margin=0.02)
BUDGET = dict(l0=120, g1=60, l1=40)
LR = dict(l0=1e-3, g1=1e-4, l1=1e-4)
MW = dict(l0=0.1, l1=0.5)
PATIENCE = dict(l0=30, g1=15, l1=10)
MIN_EP = dict(l0=0, g1=30, l1=15)
SEED = 66

LOADS = []          # provenance of every torch.load


class ContinuousF13(torch.nn.Module):
    """Explicit continuous projection: same two-parameter LS fit, no rounding."""

    @torch.no_grad()
    def forward(self, flow):
        if tuple(flow.shape[1:]) != (2, 512, 512):
            raise ValueError("expect Bx2x512x512 flow")
        yy, xx = torch.meshgrid(torch.arange(512, device=flow.device, dtype=torch.float64),
                                torch.arange(512, device=flow.device, dtype=torch.float64),
                                indexing="ij")
        x, y = xx - 256., yy - 256.
        den = (x.square() + y.square()).sum()
        f = flow.detach().double()
        a = (x * f[:, 0] + y * f[:, 1]).sum((-2, -1)) / den
        b = (x * f[:, 1] - y * f[:, 0]).sum((-2, -1)) / den
        return torch.stack([a[:, None, None] * x - b[:, None, None] * y,
                            b[:, None, None] * x + a[:, None, None] * y], 1).float()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tracked_load(path, purpose, allow):
    p = Path(path).resolve()
    ok = any(str(p).startswith(str(Path(a).resolve())) for a in allow)
    LOADS.append(dict(path=str(p), purpose=purpose, sha256=sha(p), allowed=bool(ok)))
    if not ok:
        raise RuntimeError(f"REFUSED torch.load outside allowed sources: {p} ({purpose})")
    with open(p, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def rng_restore(st):
    if not st: return
    random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


# ------------------------------------------------------------------ data
class Pairs(torch.utils.data.Dataset):
    """Cache adapter for BOTH formats, explicit dataset, no index guessing.

    os        : dict{pair_name -> float32 (2,512,512)}   (GT comes from the dataset)
    roadscene : dict{split, pairs, initial, gt, unit, shape, bridge_sha256, ckpt_sha256}
    """

    def __init__(self, split):
        cfg = machine_paths()["cache_dirs"][DATASET]
        cache_path = Path(cfg[split])
        CACHE_ROOT = Path(cfg["train"]).parent
        ck = tracked_load(cache_path, f"{split} flow cache", [CACHE_ROOT])
        self.base = plugin.make_datasets(DATASET, split, img_resize=512)
        names = [self.base[i]["pair_names"][0] for i in range(len(self.base))]
        if len(set(names)) != len(names):
            raise RuntimeError("duplicate pair names in dataset")
        self.names = names
        self.cache_path, self.cache_sha = cache_path, sha(cache_path)

        def check_shape(t, tag):
            if list(t.shape)[-3:] != [2, 512, 512]:
                raise RuntimeError(f"cache shape mismatch in {tag}: {tuple(t.shape)}")

        if DATASET == "os":
            if "initial" in ck:
                raise RuntimeError("os cache must be a plain dict{pair->flow}")
            missing = [n for n in names if n not in ck]
            if missing:
                raise RuntimeError(f"os cache missing {len(missing)} pairs, e.g. {missing[:3]}")
            check_shape(ck[names[0]], "first pair")
            self.initials = [ck[n].float() for n in names]
            self.gts = [self.base[i]["flow"].float() for i in range(len(names))]
            meta_p = cache_path.with_suffix(cache_path.suffix + ".meta.json")
            self.meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
            if self.meta.get("unit") and self.meta["unit"] != "512 native px":
                raise RuntimeError("sidecar unit mismatch")
        else:
            if ck.get("unit") != "512 native px":
                raise RuntimeError("cache unit mismatch")
            if ck.get("split") != split:
                raise RuntimeError("cache split mismatch")
            norm = lambda x: x[:-len("_visible.tif")] if x.endswith("_visible.tif") else x
            idx = {norm(q): i for i, q in enumerate(ck["pairs"])}
            if len(idx) != len(ck["pairs"]):
                raise RuntimeError("duplicate names in cache")
            want = [norm(n) for n in names]
            missing = [w for w in want if w not in idx]
            if missing:
                raise RuntimeError(f"cache missing {len(missing)} pairs, e.g. {missing[:3]}")
            check_shape(ck["initial"][0], "initial[0]")
            crft = machine_paths()["crft_ckpt_roadscene"]
            if ck.get("ckpt_sha256") and ck["ckpt_sha256"] != sha(crft):
                raise RuntimeError("cache was produced by a different CRFT checkpoint")
            self.initials = [ck["initial"][idx[w]].float() for w in want]
            self.gts = [ck["gt"][idx[w]].float() for w in want]
            self.meta = {k: ck.get(k) for k in ("unit", "shape", "bridge_sha256", "ckpt_sha256")}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        d = self.base[i]
        return dict(image0=d["image0"], image1=d["image1"], flow_init=self.initials[i],
                    flow=self.gts[i], name=self.names[i])


def geoms(item, device):
    im0, im1 = item["image0"][None], item["image1"][None]
    g = lambda t: t[0] if t.shape[1] == 1 else (0.299 * t[0][0] + 0.587 * t[0][1] + 0.114 * t[0][2])[None]
    return g(im0).unsqueeze(0).float().to(device), g(im1).unsqueeze(0).float().to(device)


def perturb_seeded(flow, s):
    st = torch.get_rng_state()
    try:
        torch.manual_seed(s); return perturb_initial(flow)
    finally:
        torch.set_rng_state(st)


def stats(v):
    v = sorted(v); n = len(v)
    med = v[n // 2] if n % 2 else .5 * (v[n // 2 - 1] + v[n // 2])
    return dict(n=n, mean=sum(v) / n, median=med, max=v[-1], lt1=sum(x < 1 for x in v),
                lt2=sum(x < 2 for x in v), ge5=sum(x >= 5 for x in v), ge10=sum(x >= 10 for x in v))


@torch.no_grad()
def val_local(model, pairs, device, limit=0):
    model.eval()
    n = len(pairs) if not limit else min(limit, len(pairs))
    out = []
    for i in range(n):
        it = pairs[i]
        g0, g1 = geoms(it, device)
        flow, _ = model(g0, g1, it["flow_init"][None].to(device), return_aux=True)
        out.append(float((flow - it["flow"][None].to(device)).norm(dim=1).mean()))
    return out


@torch.no_grad()
def val_rescue(model, pairs, device, limit=0):
    """Original run_rescue style: GlobalRescueRefiner full inference, AEPE before F13."""
    model.eval()
    n = len(pairs) if not limit else min(limit, len(pairs))
    out, sel = [], []
    for i in range(n):
        it = pairs[i]
        g0, g1 = geoms(it, device)
        flow, aux = model(g0, g1, it["flow_init"][None].to(device), return_aux=True)
        out.append(float((flow - it["flow"][None].to(device)).norm(dim=1).mean()))
        if "selection" in aux:
            sel.append(int(bool(aux["selection"][0]["selected_rescue"])))
    return out, sel


def save_ckpt(path, model, opt, sched, epoch, best_val, best_epoch, tag):
    payload = dict(state_dict=model.state_dict(), epoch=epoch, tag=tag,
                   optimizer=opt.state_dict() if opt else None,
                   scheduler=sched.state_dict() if sched else None,
                   rng=rng_state(), best_val=best_val, best_epoch=best_epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        torch.save(payload, f)


def train_local(stage, out, run, pairs, val_pairs, device, log, smoke, resume):
    set_seed(SEED)
    model = GeometryRefiner().to(device)
    if stage == "l1" and not resume:
        l0_path = out / "l0_best.ckpt"
        l0_ck = tracked_load(l0_path, "L1 initialization from this run's L0 best", [out])
        model.load_state_dict(l0_ck["state_dict"], strict=True)
        log(json.dumps(dict(
            event="l1_initialized_from_l0",
            path=str(l0_path),
            sha256=sha(l0_path)
        )))
    mw = MW[stage]
    opt = torch.optim.AdamW(model.parameters(), lr=LR[stage], weight_decay=1e-4)
    max_ep = BUDGET[stage] if not smoke else 1
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    start, best_val, best_ep, step_count = 0, float("inf"), 0, 0
    if stage == "l0" and not resume:
        init_path = out / "l0_init.pt"
        with open(init_path, "wb") as f:
            torch.save(model.state_dict(), f)
        log(json.dumps(dict(event="l0_random_init", path=str(init_path), sha256=sha(init_path),
                            note="fresh random GeometryRefiner, seed 66")))
    if resume:
        ck = tracked_load(resume, f"{stage} resume", [out])
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); rng_restore(ck["rng"])
        start, best_val, best_ep = ck["epoch"], ck["best_val"], ck["best_epoch"]
        log(f"[{stage}] resume from epoch {start}")
    rng = random.Random(SEED + start)
    for ep in range(start + 1, max_ep + 1):
        model.train()
        order = list(range(len(pairs))); rng.shuffle(order)
        t0, run_loss, nstep = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for j, i in enumerate(order):
            it = pairs[i]
            g0, g1 = geoms(it, device)
            init = it["flow_init"][None].to(device)
            if stage == "l1":
                init = perturb_seeded(init, SEED * 7919 + ep * 100003 + i)
            flow, aux = model(g0, g1, init, return_aux=True)
            loss, st = geometry_loss(aux, it["flow"][None].to(device), mw)
            (loss / 4.0).backward(); run_loss += float(loss.detach()); nstep += 1
            if (j == len(order) - 1) or ((j + 1) % 4 == 0):
                k = 4 if (j + 1) % 4 == 0 else (j % 4) + 1
                if k != 4:
                    for p in model.parameters():
                        if p.grad is not None: p.grad.mul_(4.0 / k)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step_count += 1
            if smoke and nstep >= 5:      # cross one accum boundary -> a real optimizer.step
                break
        sched.step()
        v = val_local(model, val_pairs, device, 3 if smoke else 0)
        m = sum(v) / len(v)
        improved = m < best_val - 1e-12
        if improved:
            best_val, best_ep = m, ep
        log(json.dumps(dict(stage=stage, epoch=ep, train_loss=run_loss / max(nstep, 1),
                            val_local_aepe=m, best=best_val, best_epoch=best_ep,
                            improved=improved, sec=round(time.time() - t0, 1),
                            smoke=bool(smoke))))
        save_ckpt(out / f"{stage}_last.ckpt", model, opt, sched, ep, best_val, best_ep, stage)
        if improved:
            save_ckpt(out / f"{stage}_best.ckpt", model, opt, sched, ep, best_val, best_ep, stage)
        if smoke:
            break
        if ep >= MIN_EP[stage] and (ep - best_ep) >= PATIENCE[stage]:
            log(json.dumps(dict(stage=stage, event="early_stop", epoch=ep, best_epoch=best_ep)))
            break
    return best_val, best_ep, step_count


def main():
    ap = argparse.ArgumentParser(description="RoadScene D1 independent from-scratch training")
    ap.add_argument("--stage", choices=["l0", "g1", "l1", "assemble", "all"], default="all")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--dataset", choices=["os", "roadscene"], default="roadscene")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="1 epoch / 5 steps / 8 train / 3 val")
    ap.add_argument("--resume", type=Path, default=None)
    a = ap.parse_args()
    assert a.device == "cuda" and torch.cuda.is_available(), "CUDA required"
    global DATASET
    DATASET = a.dataset
    dev = a.device
    out = (a.outdir if a.outdir.is_absolute() else ORIG_CWD / a.outdir).resolve()
    if PKG not in out.parents:
        raise SystemExit("output must live inside the project")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_log.jsonl", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); logf.write(str(msg) + "\n"); logf.flush()

    cfg = machine_paths()
    allow_load = [out, Path(cfg["cache_dirs"][DATASET]["train"]).parent,
                  Path(cfg["crft_ckpt_os"] if DATASET == "os" else cfg["crft_ckpt_roadscene"]), PKG / "src"]
    train_pairs, val_pairs, cache_info = Pairs("train"), Pairs("val"), {}
    cache_info = dict(train=dict(n=len(train_pairs), sha=train_pairs.cache_sha,
                                 path=str(train_pairs.cache_path)),
                      val=dict(n=len(val_pairs), sha=val_pairs.cache_sha,
                               path=str(val_pairs.cache_path)))
    expect = dict(os=(1172, 153), roadscene=(176, 23))[DATASET]
    if (len(train_pairs), len(val_pairs)) != expect:
        raise SystemExit(f"unexpected split sizes {len(train_pairs)}/{len(val_pairs)} for {DATASET}")
    if a.smoke:
        train_sub = torch.utils.data.Subset(train_pairs, list(range(8)))
        val_sub = torch.utils.data.Subset(val_pairs, list(range(3)))
        train_pairs, val_pairs = train_sub, val_sub

    log(json.dumps(dict(event="start", dataset=DATASET, stage=a.stage, smoke=bool(a.smoke),
                        n_train=len(train_pairs), n_val=len(val_pairs), device=dev,
                        seed=SEED, cache=cache_info,
                        crft=dict(path=str(Path(cfg["crft_ckpt_os"] if DATASET == "os" else cfg["crft_ckpt_roadscene"])),
                                  sha256=sha(Path(cfg["crft_ckpt_os"] if DATASET == "os" else cfg["crft_ckpt_roadscene"]))),
                        outdir=str(out))))

    if a.stage in ("l0", "all"):
        bv, be, steps = train_local("l0", out, None, train_pairs, val_pairs, dev, log, a.smoke, a.resume)
        log(json.dumps(dict(event="l0_done", best_val=bv, best_epoch=be, optimizer_steps=steps,
                            init_path=str(out / "l0_best.ckpt"),
                            init_sha256=sha(out / "l0_init.pt") if (out / "l0_init.pt").exists() else None)))

    if a.stage in ("g1", "all"):
        l0 = out / "l0_best.ckpt"
        ck = tracked_load(l0, "this run's L0 best", [out])
        local = GeometryRefiner().to(dev)
        local.load_state_dict(ck["state_dict"])
        # CRITICAL: build the global matcher AFTER loading L0 so its stem/shared copy L0
        d1 = GlobalRescueRefiner(local=local, **ARCH).to(dev)
        gm_sd, lo_sd = d1.global_matcher.state_dict(), local.state_dict()
        same = {k: bool(torch.equal(gm_sd[k], lo_sd[k])) for k in lo_sd if k in gm_sd}
        log(json.dumps(dict(event="g1_constructed",
                            copied_from_l0=sum(same.values()), compared=len(same),
                            all_l0_tensors_copied=all(same.values()),
                            note="GlobalMatcher deep-copies the L0 local passed in")))
        for p in d1.parameters(): p.requires_grad_(False)
        gm = list(d1.global_matcher.parameters())
        for p in gm: p.requires_grad_(True)
        set_seed(SEED)
        opt = torch.optim.AdamW(gm, lr=LR["g1"], weight_decay=1e-4)
        max_ep = BUDGET["g1"] if not a.smoke else 1
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
        best_val, best_ep, start = float("inf"), 0, 0
        if a.resume:
            r = tracked_load(a.resume, "g1 resume", [out])
            d1.load_state_dict(r["state_dict"]); opt.load_state_dict(r["optimizer"])
            sched.load_state_dict(r["scheduler"]); rng_restore(r["rng"])
            start, best_val, best_ep = r["epoch"], r["best_val"], r["best_epoch"]
        rng = random.Random(SEED + start)
        local_snapshot = {k: v.detach().clone() for k, v in d1.local.state_dict().items()}
        for ep in range(start + 1, max_ep + 1):
            d1.train(); d1.local.eval()
            order = list(range(len(train_pairs))); rng.shuffle(order)
            t0, run_loss, nstep = time.time(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for j, i in enumerate(order):
                it = train_pairs[i]
                g0, g1 = geoms(it, dev)
                rec = d1.global_matcher(g0, g1)
                loss, _ = global_match_loss(rec, it["flow"][None].to(dev))
                (loss / 4.0).backward(); run_loss += float(loss.detach()); nstep += 1
                if (j == len(order) - 1) or ((j + 1) % 4 == 0):
                    k = 4 if (j + 1) % 4 == 0 else (j % 4) + 1
                    if k != 4:
                        for p in gm:
                            if p.grad is not None: p.grad.mul_(4.0 / k)
                    torch.nn.utils.clip_grad_norm_(gm, 1.0)
                    opt.step(); opt.zero_grad(set_to_none=True)
                if a.smoke and nstep >= 5:
                    break
            sched.step()
            v, sel = val_rescue(d1, val_pairs, dev, 3 if a.smoke else 0)
            m = sum(v) / len(v)
            improved = m < best_val - 1e-12
            if improved: best_val, best_ep = m, ep
            local_unchanged = all(torch.equal(local_snapshot[k], v2)
                                  for k, v2 in d1.local.state_dict().items())
            log(json.dumps(dict(stage="g1", epoch=ep, train_loss=run_loss / max(nstep, 1),
                                val_globalrescue_aepe=m, best=best_val, best_epoch=best_ep,
                                rescue_used=sum(sel), local_unchanged=local_unchanged,
                                improved=improved, sec=round(time.time() - t0, 1), smoke=bool(a.smoke))))
            save_ckpt(out / "g1_last.ckpt", d1, opt, sched, ep, best_val, best_ep, "g1")
            if improved:
                save_ckpt(out / "g1_best.ckpt", d1, opt, sched, ep, best_val, best_ep, "g1")
            if a.smoke: break
            if ep >= MIN_EP["g1"] and (ep - best_ep) >= PATIENCE["g1"]:
                log(json.dumps(dict(stage="g1", event="early_stop", epoch=ep, best_epoch=best_ep)))
                break

    if a.stage in ("l1", "all"):
        bv, be, steps = train_local("l1", out, None, train_pairs, val_pairs, dev, log, a.smoke, a.resume)
        log(json.dumps(dict(event="l1_done", best_val=bv, best_epoch=be, optimizer_steps=steps)))

    if a.stage in ("assemble", "all"):
        l1 = tracked_load(out / "l1_best.ckpt", "this run's L1 best", [out])
        g1 = tracked_load(out / "g1_best.ckpt", "this run's G1 best", [out])
        local = GeometryRefiner().to(dev)
        local.load_state_dict(l1["state_dict"])
        d1 = DiverseRescueRefiner(local=local, **ARCH).to(dev)
        sd = d1.state_dict()
        for k in sd:
            if k.startswith("global_matcher."):
                sd[k] = g1["state_dict"][k]
        missing = [k for k in sd if k not in g1["state_dict"] and k.startswith("global_matcher.")]
        if missing:
            raise RuntimeError(f"missing global tensors: {missing[:3]}")
        d1.load_state_dict(sd, strict=True)
        local_ok = all(torch.equal(sd[f"local.{k}"], v) for k, v in d1.local.state_dict().items())
        payload = dict(kind="frozen_d1_v1", architecture=ARCH, state_dict=d1.state_dict(),
                       protocol=dict(dataset=DATASET, from_scratch=True, seed=SEED,
                                     stages="random L0 -> G1/L1 -> assemble",
                                     l0_sha256=sha(out / "l0_best.ckpt"),
                                     g1_sha256=sha(out / "g1_best.ckpt"),
                                     l1_sha256=sha(out / "l1_best.ckpt"),
                                     note="RoadScene plug-in trained without any OS/past plug-in weights"))
        with open(out / "assembled_d1.ckpt", "wb") as f:
            torch.save(payload, f)
        # final report check (continuous F13) on full val
        prior = ContinuousF13().to(dev)
        d1.eval().requires_grad_(False)
        rows = []
        val_full = Pairs("val")
        for i in range(len(val_full)):
            it = val_full[i]
            g0, g1t = geoms(it, dev)
            with torch.no_grad():
                out_flow, aux = d1(g0, g1t, it["flow_init"][None].to(dev), return_aux=True)
                cont = prior(out_flow)
                gt = it["flow"][None].to(dev)
                rows.append(dict(pair=it["name"],
                                 bridge=float((it["flow_init"][None].to(dev) - gt).norm(dim=1).mean()),
                                 d1=float((out_flow - gt).norm(dim=1).mean()),
                                 d1_continuous=float((cont - gt).norm(dim=1).mean()),
                                 selected_rescue=int(bool(aux["selection"][0]["selected_rescue"]))))
        log(json.dumps(dict(event="assemble_done", assembled_sha256=sha(out / "assembled_d1.ckpt"),
                            local_from_l1_ok=local_ok,
                            d1=stats([r["d1"] for r in rows]),
                            d1_continuous=stats([r["d1_continuous"] for r in rows]),
                            rescue_used=sum(r["selected_rescue"] for r in rows)), ensure_ascii=False))
        (out / "final_val_assembled.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    (out / "load_provenance.json").write_text(json.dumps(LOADS, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps(dict(event="loads", n=len(LOADS), all_allowed=all(x["allowed"] for x in LOADS))))
    logf.close()


if __name__ == "__main__":
    main()
