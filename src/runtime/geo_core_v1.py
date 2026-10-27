"""Image-conditioned similarity-flow plugin. No CRFT imports or parameters.

Inputs/outputs are Bx2xHxW flows in native image pixel units.
The initial flow is projected to a four-parameter similarity model; therefore
this model is NOT initially an identity plugin. Bypass the module for `off`.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def sample_at(image, xy):
    """image B,C,H,W; xy B,...,2 in image-pixel units -> B,C,... ."""
    b,c,h,w=image.shape
    shape=xy.shape[1:-1]
    grid=2*(xy+0.5)/xy.new_tensor([w,h])-1
    out=F.grid_sample(image,grid.reshape(b,1,-1,2),mode='bilinear',
                      padding_mode='zeros',align_corners=False)
    return out.reshape(b,c,*shape)


def anchor_grid(h,w,side,device,dtype):
    y=(torch.arange(side,device=device,dtype=dtype)+.5)*h/side-.5
    x=(torch.arange(side,device=device,dtype=dtype)+.5)*w/side-.5
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    return torch.stack([xx,yy],-1).reshape(1,-1,2)


def similarity_design(points,h,w):
    # Isotropic normalization, so the four coefficients really are similarity.
    scale=max(h-1,w-1)/2
    p=(points-points.new_tensor([(w-1)/2,(h-1)/2]))/scale
    x,y=p.unbind(-1); z=torch.zeros_like(x); o=torch.ones_like(x)
    return torch.stack([torch.stack([x,-y,o,z],-1),
                        torch.stack([y,x,z,o],-1)],-2)


def similarity_flow(theta,points,h,w):
    return torch.einsum('bmck,bk->bmc',similarity_design(points,h,w),theta)


def fit_similarity(points,displacements,weights,h,w,prior=None,ridge=1e-6):
    """Differentiable 4x4 weighted solve; fallback for insufficient geometry.

    No GT input. weights B,M; displacement B,M,2. Computation stays fp32/64.
    A ridge toward `prior` avoids singular solves; low-rank/low-count samples
    return prior instead of extrapolating unsupported parameters.
    """
    d=similarity_design(points,h,w)
    if d.shape[0]!=displacements.shape[0]: d=d.expand(displacements.shape[0],-1,-1,-1)
    weights=weights.clamp_min(0)
    wn=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
    normal=torch.einsum('bmci,bm,bmcj->bij',d,wn,d)
    rhs=torch.einsum('bmci,bm,bmc->bi',d,wn,displacements)
    if prior is None: prior=rhs.new_zeros(rhs.shape)
    eye=torch.eye(4,device=rhs.device,dtype=rhs.dtype)[None]
    result=torch.linalg.solve(normal+ridge*eye,(rhs+ridge*prior).unsqueeze(-1)).squeeze(-1)
    good=((weights>1e-6).sum(-1)>=4)&(torch.linalg.eigvalsh(normal.detach())[:,0]>1e-5)
    return torch.where(good[:,None],result,prior)


def dense_similarity(theta,h,w):
    y,x=torch.meshgrid(torch.arange(h,device=theta.device,dtype=theta.dtype),
                        torch.arange(w,device=theta.device,dtype=theta.dtype),indexing='ij')
    p=torch.stack([x,y],-1).reshape(1,-1,2).expand(len(theta),-1,-1)
    return similarity_flow(theta,p,h,w).transpose(1,2).reshape(-1,2,h,w)


def standardize(x):
    return (x-x.mean((-2,-1),keepdim=True))/(x.std((-2,-1),keepdim=True)+1e-6)


class Block(nn.Sequential):
    def __init__(self,ci,co,dilation=1):
        super().__init__(nn.Conv2d(ci,co,3,padding=dilation,dilation=dilation,bias=False),
                         nn.GroupNorm(8,co),nn.GELU())


class MatchStage(nn.Module):
    def __init__(self,channels,radius,anchors):
        super().__init__(); self.radius=radius; self.anchors=anchors
        y,x=torch.meshgrid(torch.arange(-radius,radius+1),torch.arange(-radius,radius+1),indexing='ij')
        self.register_buffer('offsets',torch.stack([x,y],-1).float().reshape(-1,2))
        k=len(self.offsets)
        self.log_temperature=nn.Parameter(torch.tensor(math.log(.10)))
        self.correction=nn.Sequential(Block(2*k+channels+4,64),Block(64,64),nn.Conv2d(64,k+1,1))
        nn.init.zeros_(self.correction[-1].weight); nn.init.zeros_(self.correction[-1].bias)

    def forward(self,f0,f1,theta,points,h,w):
        b,c,gh,gw=f0.shape; m=points.shape[1]; k=len(self.offsets)
        flow=similarity_flow(theta,points,h,w)
        center=points+flow
        ratio=points.new_tensor([gw/w,gh/h])
        pfeat=(points+.5)*ratio-.5
        qfeat=(center+.5)*ratio-.5
        candidates=qfeat[:,:,None,:]+self.offsets[None,None,:,:]
        valid=(candidates[...,0]>=0)&(candidates[...,0]<=gw-1)&(candidates[...,1]>=0)&(candidates[...,1]<=gh-1)
        source=F.normalize(sample_at(f0,pfeat),dim=1,eps=1e-6)
        # Sparse anchor grid avoids a full-image all-pairs correlation tensor.
        target=F.normalize(sample_at(f1,candidates),dim=1,eps=1e-6)
        corr=(source[:,:,:,None]*target).sum(1)
        temperature=self.log_temperature.exp().clamp(.02,.5)
        logits=corr/temperature
        coord=(points/points.new_tensor([w,h])-.5).transpose(1,2)
        evidence=torch.cat([logits.transpose(1,2),valid.float().transpose(1,2),
                            source,flow.transpose(1,2)/16,coord],1)
        correction=self.correction(evidence.reshape(b,-1,self.anchors,self.anchors)).reshape(b,k+1,m)
        logits=logits+correction[:,:k].transpose(1,2)
        any_valid=valid.any(-1)
        # Fully out-of-frame anchors produce zero weight, no softmax NaNs.
        dummy=torch.zeros_like(valid); dummy[...,k//2]=True
        safe_valid=torch.where(any_valid[:,:,None],valid,dummy)
        masked=logits.masked_fill(~safe_valid,-1e4)
        prob=masked.softmax(-1)
        delta=torch.einsum('bmk,kc->bmc',prob,self.offsets)/ratio
        confidence=(.1+.9*correction[:,k].sigmoid())*any_valid.float()
        proposals=center+delta
        updated=fit_similarity(points,proposals-points,confidence,h,w,prior=theta)
        return updated,dict(logits=masked,valid=valid,centers=center,points=points,
                            offsets=self.offsets,ratio=ratio,confidence=confidence)


class GeometryRefiner(nn.Module):
    def __init__(self,in_img_ch=1,base_ch=32,anchors=24,grids=(128,256),radii=(6,4)):
        super().__init__()
        if base_ch%8 or len(grids)!=len(radii) or anchors<2: raise ValueError('Invalid architecture')
        self.config=dict(in_img_ch=in_img_ch,base_ch=base_ch,anchors=anchors,grids=tuple(grids),radii=tuple(radii))
        self.anchors=anchors; self.grids=tuple(grids)
        # Modality-specific first layers; shared structural feature layers.
        self.stem0=Block(in_img_ch,base_ch); self.stem1=Block(in_img_ch,base_ch)
        self.shared=nn.Sequential(Block(base_ch,base_ch),Block(base_ch,base_ch,2),nn.Conv2d(base_ch,base_ch,1))
        self.stages=nn.ModuleList([MatchStage(base_ch,r,anchors) for r in radii])

    def forward(self,img0,img1,flow_init,return_aux=False,detach_initial=True):
        if img0.shape!=img1.shape or img0.shape[0]!=flow_init.shape[0] or img0.shape[-2:]!=flow_init.shape[-2:]:
            raise ValueError('Images and flow must share batch and spatial size')
        b,_,h,w=flow_init.shape
        if h!=w: raise ValueError('This experimental feature pyramid expects square images')
        # Inference keeps its original boundary; joint training opts into live flow.
        f0=flow_init.detach() if detach_initial else flow_init
        points=anchor_grid(h,w,self.anchors,f0.device,f0.dtype).expand(b,-1,-1)
        init_vectors=sample_at(f0,points).transpose(1,2)
        weights=f0.new_ones((b,points.shape[1]))
        theta=fit_similarity(points,init_vectors,weights,h,w)
        initial=dense_similarity(theta,h,w)
        outputs=[]; records=[]
        i0=standardize(img0.detach()); i1=standardize(img1.detach())
        for grid,stage in zip(self.grids,self.stages):
            # Both image and point resize use the same half-pixel convention.
            a=F.interpolate(i0,(grid,grid),mode='bilinear',align_corners=False)
            bimg=F.interpolate(i1,(grid,grid),mode='bilinear',align_corners=False)
            a=self.shared(self.stem0(a)); bimg=self.shared(self.stem1(bimg))
            theta,record=stage(a,bimg,theta,points,h,w)
            outputs.append(dense_similarity(theta,h,w)); records.append(record)
        if return_aux: return outputs[-1],dict(sequence=outputs,matches=records,initial_projection=initial,theta=theta)
        return outputs[-1]


def geometry_loss(aux,gt,match_weight=.1,gamma=.8):
    """Full-image endpoint loss + in-search supervised correspondence loss.

    GT masks apply ONLY to auxiliary matching supervision, never AEPE.
    """
    sequence=aux['sequence']; weights=[gamma**(len(sequence)-1-i) for i in range(len(sequence))]
    dense=sum(w*torch.sqrt((out-gt).square().sum(1)+1e-6).mean() for w,out in zip(weights,sequence))/sum(weights)
    ce=gt.sum()*0; cover=[]
    for rec in aux['matches']:
        # Matching targets are labels: do not backpropagate into their centers.
        target=sample_at(gt,rec['points']).transpose(1,2)
        truth_q=rec['points']+target
        offset=((truth_q-rec['centers'].detach())*rec['ratio']).detach()
        r=rec['offsets'].abs().max()
        h,w=gt.shape[-2:]
        inside=(truth_q[...,0]>=0)&(truth_q[...,0]<=w-1)&(truth_q[...,1]>=0)&(truth_q[...,1]<=h-1)
        eligible=inside&(offset.abs().amax(-1)<=r)&rec['valid'].any(-1)
        distances=(offset[:,:,None]-rec['offsets'][None,None]).square().sum(-1)
        labels=(-distances/(2*.75**2)).masked_fill(~rec['valid'],-1e4).softmax(-1).detach()
        per=-(labels*rec['logits'].log_softmax(-1)).sum(-1)
        ce=ce+(per*eligible).sum()/eligible.sum().clamp_min(1)
        cover.append(eligible.float().mean().detach())
    ce=ce/len(aux['matches'])
    return dense+match_weight*ce,dict(endpoint=dense.detach(),match=ce.detach(),coverage=torch.stack(cover).mean())
