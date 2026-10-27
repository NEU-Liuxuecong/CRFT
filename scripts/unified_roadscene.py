"""Unified RoadScene inference: CRFT -> Bridge -> D1 -> Continuous F13."""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "runtime"))
sys.path.insert(0, str(ROOT / "scripts"))

import plugin
from diverse_rescue import DiverseRescueRefiner
from train_independent import ContinuousF13


class UnifiedCRFTD1F13(nn.Module):
    def __init__(self, ckpt_path, device="cuda"):
        super().__init__()

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path

        payload = torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )
        if payload["dataset"] != "roadscene":
            raise ValueError("Expected RoadScene checkpoint")

        self.device_name = device

        # Construct CRFT without loading an external checkpoint.
        cfg = plugin.bridge.build_config("roadscene")
        self.crft = plugin.bridge.PL_CRFT(cfg, pretrained_ckpt=None)

        matcher_state = {
            k.removeprefix("matcher."): v
            for k, v in payload["crft_state_dict"].items()
            if k.startswith("matcher.")
        }
        self.crft.matcher.load_state_dict(matcher_state, strict=True)

        self.d1 = DiverseRescueRefiner(
            **payload["d1_architecture"]
        )
        self.d1.load_state_dict(
            payload["d1_state_dict"], strict=True
        )

        self.f13 = ContinuousF13()

        self.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, small_item, native_item):
        """Accept the same 64/512 dataset items as the original pipeline."""
        batch = default_collate([
            {k: v for k, v in small_item.items() if k != "flow"}
        ])

        initial = plugin.bridge.flow_to_512(
            plugin.bridge.backbone_flow(
                self.crft, batch, device=self.device_name
            )
        )

        g0 = native_item["image0"][None].to(self.device_name).float()
        g1 = native_item["image1"][None].to(self.device_name).float()

        # Match train_independent.geoms() exactly for RoadScene RGB inputs.
        def gray(t):
            if t.shape[1] == 1:
                return t
            if t.shape[1] != 3:
                raise ValueError("Expected 1 or 3 image channels")
            return (0.299 * t[:, 0] + 0.587 * t[:, 1]
                    + 0.114 * t[:, 2]).unsqueeze(1)

        g0, g1 = gray(g0), gray(g1)

        pred = self.d1(g0, g1, initial.float())
        if isinstance(pred, tuple):
            pred = pred[0]

        return self.f13(pred)

    @torch.no_grad()
    def predict(self, visible_path, infrared_path):
        """Predict 512x512 RoadScene flow directly from two image paths."""
        import torch.nn.functional as F
        from src.utils.dataset import read_image_rgb

        visible_path = str(Path(visible_path).resolve())
        infrared_path = str(Path(infrared_path).resolve())

        def prepare(resolution):
            image0, mask0, scale0 = read_image_rgb(
                visible_path, resolution, 8, True, augment_fn=None
            )
            image1, mask1, scale1 = read_image_rgb(
                infrared_path, resolution, 8, True, augment_fn=None
            )

            if image0.shape[-2:] != (resolution, resolution):
                raise ValueError(f"Unexpected visible image shape: {image0.shape}")
            if image1.shape[-2:] != (resolution, resolution):
                raise ValueError(f"Unexpected infrared image shape: {image1.shape}")

            data = {
                "image0": image0,
                "image1": image1,
                "scale0": scale0,
                "scale1": scale1,
                "dataset_name": "RoadScence",
                "pair_id": 0,
                "pair_names": (
                    Path(visible_path).name,
                    Path(infrared_path).name
                ),
                "image0_path": visible_path,
                "image1_path": infrared_path
            }

            if mask0 is not None:
                masks = F.interpolate(
                    torch.stack([mask0, mask1], dim=0)[None].float(),
                    scale_factor=0.125,
                    mode="nearest",
                    recompute_scale_factor=False
                )[0].bool()

                data["mask0"] = masks[0]
                data["mask1"] = masks[1]

            return data

        small_item = prepare(64)
        native_item = prepare(512)

        return self.forward(small_item, native_item)
