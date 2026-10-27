"""Training entry point for CRFT-based image registration."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="CRFT Training")

    parser.add_argument(
        "--dataset",
        choices=["os", "roadscene"],
        default="os",
    )
    parser.add_argument(
        "--stage",
        choices=["l0", "g1", "l1", "assemble", "all"],
        default="all",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")

    args = parser.parse_args()

    outdir = args.outdir or (
        ROOT / "runs" / f"{args.dataset}_repro"
    )

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_independent.py"),
        "--dataset", args.dataset,
        "--stage", args.stage,
        "--outdir", str(outdir),
        "--device", args.device,
    ]

    if args.smoke:
        command.append("--smoke")

    print("Training command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
