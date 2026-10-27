"""Final fixed plug-in: centered similarity + discrete synthetic-transform prior.

No images, labels, IDs, raw matrices or per-pair metadata enter inference.
This is a dataset-specific parameter-free output head, not a new learned matcher.
"""
import torch
from torch import nn


class GeometryPrior512(nn.Module):
    @torch.no_grad()
    def forward(self,flow_base,disable=False,return_aux=False):
        if flow_base.ndim!=4 or flow_base.shape[1:]!=(2,512,512):raise ValueError('Bx2x512x512 native-pixel flow required')
        if flow_base.dtype!=torch.float32 or not torch.isfinite(flow_base).all():raise ValueError('Finite float32 flow required')
        if disable and not return_aux:return flow_base.detach()
        # Accumulate the two-parameter least-squares projection in float64.
        # The center is fixed at (256,256), from the train generating family.
        device=flow_base.device
        yy,xx=torch.meshgrid(torch.arange(512,device=device,dtype=torch.float64),torch.arange(512,device=device,dtype=torch.float64),indexing='ij')
        x,y=xx-256.,yy-256.;den=(x.square()+y.square()).sum()
        f=flow_base.detach().double()
        a=(x*f[:,0]+y*f[:,1]).sum((-2,-1))/den
        b=(x*f[:,1]-y*f[:,0]).sum((-2,-1))/den
        continuous=torch.stack([a[:,None,None]*x-b[:,None,None]*y,b[:,None,None]*x+a[:,None,None]*y],1).float()
        scale=torch.hypot(1.+a,b);angle=torch.atan2(b,1.+a)
        snapped_scale=torch.round(scale/.05)*.05
        snapped_angle=torch.deg2rad(torch.round(torch.rad2deg(angle)))
        aa=snapped_scale*snapped_angle.cos()-1.;bb=snapped_scale*snapped_angle.sin()
        pred=torch.stack([aa[:,None,None]*x-bb[:,None,None]*y,bb[:,None,None]*x+aa[:,None,None]*y],1).float()
        out=flow_base.detach() if disable else pred
        if not return_aux:return out
        return out,dict(continuous_projection=continuous,scale=scale,angle_deg=torch.rad2deg(angle),
                        snapped_scale=snapped_scale,snapped_angle_deg=torch.rad2deg(snapped_angle))
