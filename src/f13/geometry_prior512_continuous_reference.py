"""F13 ablation: retain centered least-squares projection; remove both roundings.

Same 512 input, center (256,256), precision and no-independent-translation family.
Output is exactly the original F13's continuous_projection diagnostic tensor.
No images, GT, IDs or per-pair metadata enter inference. No learned parameters.
"""
import torch
from torch import nn


class GeometryPrior512Continuous(nn.Module):
    @torch.no_grad()
    def forward(self, flow_base, disable=False, return_aux=False):
        if flow_base.ndim != 4 or flow_base.shape[1:] != (2, 512, 512):
            raise ValueError('Bx2x512x512 native-pixel flow required')
        if flow_base.dtype != torch.float32 or not torch.isfinite(flow_base).all():
            raise ValueError('Finite float32 flow required')
        if disable and not return_aux:
            return flow_base.detach()
        device = flow_base.device
        yy, xx = torch.meshgrid(
            torch.arange(512, device=device, dtype=torch.float64),
            torch.arange(512, device=device, dtype=torch.float64), indexing='ij')
        x, y = xx - 256., yy - 256.
        den = (x.square() + y.square()).sum()
        f = flow_base.detach().double()
        a = (x*f[:, 0] + y*f[:, 1]).sum((-2, -1)) / den
        b = (x*f[:, 1] - y*f[:, 0]).sum((-2, -1)) / den
        continuous = torch.stack([
            a[:, None, None]*x - b[:, None, None]*y,
            b[:, None, None]*x + a[:, None, None]*y], 1).float()
        # Keep continuous parameters for inspection. No round, grid or snapping.
        scale = torch.hypot(1. + a, b)
        angle = torch.atan2(b, 1. + a)
        out = flow_base.detach() if disable else continuous
        if not return_aux:
            return out
        return out, dict(continuous_projection=continuous, scale=scale,
                         angle_deg=torch.rad2deg(angle))
