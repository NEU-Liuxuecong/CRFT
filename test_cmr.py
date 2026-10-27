"""Evaluate AEPE and CMR on the complete native-resolution dataset."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import plugin
from inference import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["os", "roadscene"], default="roadscene")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_class, _ = MODELS[args.dataset]
    model = model_class(args.ckpt.resolve(), device=args.device)

    small_dataset = plugin.make_datasets(args.dataset, args.split, img_resize=64)
    native_dataset = plugin.make_datasets(args.dataset, args.split, img_resize=512)

    assert len(small_dataset) == len(native_dataset)

    results = []
    thresholds = (3.0, 1.0, 0.7)

    with torch.inference_mode():
        for index in range(len(native_dataset)):
            small_item = small_dataset[index]
            native_item = native_dataset[index]

            assert tuple(small_item["pair_names"]) == tuple(native_item["pair_names"])

            pred = model(small_item, native_item)
            gt = native_item["flow"][None].to(args.device).float()

            assert pred.shape == gt.shape

            epe = torch.linalg.vector_norm(pred - gt, ord=2, dim=1)

            if not torch.isfinite(epe).all():
                raise RuntimeError(f"Non-finite EPE at sample {index}")

            item = {
                "index": index,
                "pair": str(native_item["pair_names"][0]),
                "AEPE": epe.mean().item(),
            }

            for threshold in thresholds:
                item[f"CMR@{threshold:g}px"] = (
                    (epe < threshold).float().mean().item() * 100.0
                )

            results.append(item)

            print(
                f"[{index+1:02d}/{len(native_dataset)}] "
                f"AEPE={item['AEPE']:.4f}  "
                f"CMR@3={item['CMR@3px']:.2f}%  "
                f"CMR@1={item['CMR@1px']:.2f}%  "
                f"CMR@0.7={item['CMR@0.7px']:.2f}%",
                flush=True,
            )

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "count": len(results),
        "checkpoint": str(args.ckpt.resolve()),
        "AEPE": sum(x["AEPE"] for x in results) / len(results),
    }

    for threshold in thresholds:
        key = f"CMR@{threshold:g}px"
        summary[key] = sum(x[key] for x in results) / len(results)

    output = args.output or (ROOT / "results" / "cmr_evaluation")
    output.mkdir(parents=True, exist_ok=True)

    with (output / "per_pair.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== AEPE & CMR Results ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
