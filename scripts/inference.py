"""Unified inference entry point for OSdataset and RoadScene."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from unified_os import UnifiedCRFTD1F13 as OSModel
from unified_roadscene import UnifiedCRFTD1F13 as RoadSceneModel


MODELS = {
    "os": (
        OSModel,
        ROOT / "weights/os/CRFT_D1_F13_OS_best.ckpt"
    ),
    "roadscene": (
        RoadSceneModel,
        ROOT / "weights/roadscene/CRFT_D1_F13_RoadScene_best.ckpt"
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="CRFT + D1 + Continuous F13 image registration"
    )

    parser.add_argument(
        "--dataset",
        choices=["os", "roadscene"],
        required=True
    )
    parser.add_argument("--image0", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    image0 = Path(args.image0).expanduser().resolve()
    image1 = Path(args.image1).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    for path in (image0, image1):
        if not path.is_file():
            parser.error(f"Image not found: {path}")

    model_class, checkpoint = MODELS[args.dataset]

    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Checkpoint: {checkpoint}")

    model = model_class(checkpoint, device=args.device)

    with torch.inference_mode():
        flow = model.predict(image0, image1)

    if flow.shape != (1, 2, 512, 512):
        raise RuntimeError(
            f"Unexpected output shape: {tuple(flow.shape)}"
        )

    result = flow[0].detach().cpu().numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result)

    print(f"Saved: {output}")
    print(f"Flow shape: {result.shape}")
    print(f"Flow dtype: {result.dtype}")


if __name__ == "__main__":
    main()
