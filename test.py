"""Evaluate pretrained unified CRFT models on OSdataset or RoadScene."""

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
    parser = argparse.ArgumentParser(description="CRFT Evaluation")
    parser.add_argument(
        "--dataset",
        choices=["os", "roadscene"],
        default="os",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Optional unified model checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means evaluate the complete split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")

    model_class, default_checkpoint = MODELS[args.dataset]
    checkpoint = (
        args.ckpt.expanduser().resolve()
        if args.ckpt is not None
        else default_checkpoint
    )

    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    model = model_class(checkpoint, device=args.device)

    small_dataset = plugin.make_datasets(
        args.dataset, args.split, img_resize=64
    )
    native_dataset = plugin.make_datasets(
        args.dataset, args.split, img_resize=512
    )

    if len(small_dataset) != len(native_dataset):
        raise RuntimeError("Dataset length mismatch")

    count = len(native_dataset)
    if args.limit:
        count = min(count, args.limit)

    errors = []

    with torch.inference_mode():
        for index in range(count):
            small_item = small_dataset[index]
            native_item = native_dataset[index]

            if tuple(small_item["pair_names"]) != tuple(native_item["pair_names"]):
                raise RuntimeError(f"Pair mismatch at index {index}")

            pred = model(small_item, native_item)
            gt = native_item["flow"][None].to(args.device).float()

            if pred.shape != gt.shape:
                raise RuntimeError(
                    f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"
                )

            aepe = torch.linalg.vector_norm(
                pred - gt, ord=2, dim=1
            ).mean().item()

            errors.append({
                "index": index,
                "pair": str(native_item["pair_names"][0]),
                "aepe": aepe,
            })

            print(
                f"[{index + 1}/{count}] "
                f"AEPE={aepe:.6f}",
                flush=True,
            )

    mean_aepe = sum(item["aepe"] for item in errors) / len(errors)

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "count": count,
        "checkpoint": str(checkpoint),
        "AEPE": mean_aepe,
    }

    output = args.output or (
        ROOT / "results" / f"{args.dataset}_{args.split}"
    )
    output.mkdir(parents=True, exist_ok=True)

    with (output / "per_pair.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["index", "pair", "aepe"]
        )
        writer.writeheader()
        writer.writerows(errors)

    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== Evaluation Results ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
