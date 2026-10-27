"""Post-CRFT global similarity rescue. Forward NEVER accepts ground truth.

The old local plugin is preserved as an independent branch. Global descriptors
are an independent copy: global training cannot alter the local branch.
Matching/consensus/selection use images and predictions only. No sample IDs.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from geo_core_v1 import (GeometryRefiner, Block, anchor_grid, sample_at, standardize,
                         fit_similarity, similarity_flow, dense_similarity)


class GlobalMatcher(nn.Module):
    def __init__(self, local, grid=64, anchors=24):
        super().__init__()
        self.stem0=copy.deepcopy(local.stem0)
        self.stem1=copy.deepcopy(local.stem1)
        self.shared=copy.deepcopy(local.shared)
        channels=local.config['base_ch']
        self.context=nn.Sequential(Block(channels,64),Block(64,64,2),Block(64,64,4),nn.Conv2d(64,channels,1))
        nn.init.zeros_(self.context[-1].weight); nn.init.zeros_(self.context[-1].bias)
        self.log_temperature=nn.Parameter(torch.tensor(math.log(.10)))
        self.grid=grid; self.anchors=anchors

    def forward(self, image0, image1):
        b,_,h,w=image0.shape
        features=[]
        for image,stem in ((image0,self.stem0),(image1,self.stem1)):
            # Start with the same image resolution as old stage 1.
            small=F.interpolate(standardize(image.detach()),(128,128),mode='bilinear',align_corners=False)
            feature=self.shared(stem(small))
            features.append(feature+self.context(feature))
        points=anchor_grid(h,w,self.anchors,image0.device,image0.dtype).expand(b,-1,-1)
        targets=anchor_grid(h,w,self.grid,image0.device,image0.dtype).expand(b,-1,-1)
        ratio=points.new_tensor([128/w,128/h])
        a=F.normalize(sample_at(features[0],(points+.5)*ratio-.5),dim=1,eps=1e-6)
        z=F.normalize(sample_at(features[1],(targets+.5)*ratio-.5),dim=1,eps=1e-6)
        logits=torch.einsum('bcm,bcn->bmn',a,z)/self.log_temperature.exp().clamp(.02,.5)
        return dict(logits=logits,points=points,targets=targets,grid=self.grid)


def global_match_loss(record, gt):
    """GT labels only during TRAINING: full-target soft CE + source competition.

    Source-grid competition is supervised only near a visible GT match. It
    does not invent correspondences for non-overlap/out-of-frame points.
    """
    logits=record['logits']; points=record['points']; targets=record['targets']
    h,w=gt.shape[-2:]; grid=record['grid']
    truth=points+sample_at(gt.detach(),points).transpose(1,2)
    visible=(truth[...,0]>=0)&(truth[...,0]<=w-1)&(truth[...,1]>=0)&(truth[...,1]<=h-1)
    distance=((truth[:,:,None]-targets[:,None])*truth.new_tensor([grid/w,grid/h])).square().sum(-1)
    weights=torch.exp(-distance/(2*.75**2))*visible[:,:,None]
    row_target=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    forward=(-(row_target*logits.log_softmax(-1)).sum(-1)*visible).sum()/visible.sum().clamp_min(1)
    close=weights.amax(1)>.1
    col_target=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
    reverse=(-(col_target*logits.log_softmax(1)).sum(1)*close).sum()/close.sum().clamp_min(1)
    peak=targets.gather(1,logits.argmax(-1)[...,None].expand(-1,-1,2))
    peak_error=((peak-truth).norm(dim=-1)*visible).sum()/visible.sum().clamp_min(1)
    return forward+.5*reverse,dict(forward_ce=forward.detach(),reverse_ce=reverse.detach(),
                                  visible=visible.float().mean().detach(),peak_error=peak_error.detach())


def two_point_theta(p, q, h, w):
    """N,2,2 points -> N,4 normalized similarity displacement parameters."""
    dp=p[:,1]-p[:,0]; dq=q[:,1]-q[:,0]
    denominator=dp.square().sum(-1).clamp_min(1e-8)
    a=(dp*dq).sum(-1)/denominator
    b=(dp[:,0]*dq[:,1]-dp[:,1]*dq[:,0])/denominator
    c=p.new_tensor([(w-1)/2,(h-1)/2]); s=max(h-1,w-1)/2
    u=p[:,0]-c
    shift=q[:,0]-c-torch.stack([a*u[:,0]-b*u[:,1],b*u[:,0]+a*u[:,1]],-1)
    theta=torch.stack([(a-1)*s,b*s,shift[:,0],shift[:,1]],-1)
    valid=(denominator>=(min(h,w)/8)**2)&(a.square()+b.square()>=.5**2)&(a.square()+b.square()<=2**2)
    return theta,valid


def correspondence_set(record):
    # Dual softmax discourages many sources choosing one popular target.
    logits=record['logits']
    dual=(logits.log_softmax(-1)+logits.log_softmax(1)).exp()
    confidence,index=dual.topk(2,dim=-1)
    targets=record['targets']
    q=targets[:,None].expand(-1,index.shape[1],-1,-1).gather(2,index[...,None].expand(-1,-1,-1,2))
    # Bound domination by a handful of very high confidence correspondences.
    weights=(confidence/confidence[:,:,0].mean(-1)[:,None,None].clamp_min(1e-8)).clamp(max=3.)
    return record['points'],q,weights


def consensus(theta, points, q, weights, h, w, sigma):
    """H hypotheses on one image; returns one score/count/spread per H."""
    projected=points+similarity_flow(theta,points,h,w)
    distance=(projected[:,:,None]-q).norm(dim=-1)
    evidence=weights*torch.exp(-.5*(distance/sigma).square())
    per,choice=evidence.max(-1)
    score=per.sum(-1)/weights.amax(-1).sum(-1).clamp_min(1e-8)
    nearest=distance.gather(-1,choice[...,None]).squeeze(-1)
    inliers=nearest<=sigma
    # Require inliers spanning several source-image cells.
    cell=(points[...,0]*4/w).long().clamp(0,3)+4*(points[...,1]*4/h).long().clamp(0,3)
    coverage=torch.stack([(inliers&(cell==i)).any(-1) for i in range(16)],-1).sum(-1)
    return score,inliers.sum(-1),coverage,choice,nearest


@torch.no_grad()
def robust_pose(record, prior, h, w, hypotheses=1024):
    points,queries,weights=correspondence_set(record)
    sigma=12.*max(h,w)/512
    results=[]
    for b in range(len(prior)):
        p=points[b:b+1];q=queries[b:b+1];wt=weights[b:b+1]
        flat_weights=wt.flatten()
        count=min(256,len(flat_weights))
        ids=flat_weights.topk(count).indices
        # Fixed sampling schedule, independent of validation IDs or GT.
        generator=torch.Generator(device='cpu').manual_seed(1931)
        pairs=torch.randint(count,(hypotheses,2),generator=generator).to(p.device)
        selected=ids[pairs]
        pp=p[0,selected//2];qq=q[0,selected//2,selected%2]
        candidates,valid=two_point_theta(pp,qq,h,w)
        candidates=candidates[valid]
        candidates=torch.cat([prior[b:b+1],candidates],0)
        scores=[]
        for chunk in candidates.split(128):
            sc,n,spread,_,_=consensus(chunk,p,q,wt,h,w,sigma)
            sc=sc.masked_fill((n<max(12,math.ceil(p.shape[1]*.05)))|(spread<6),-1)
            scores.append(sc)
        all_scores=torch.cat(scores)
        theta=candidates[all_scores.argmax()][None]
        if all_scores.max()<0:
            results.append(prior[b]);continue
        for _ in range(3):
            _,_,_,choice,distance=consensus(theta,p,q,wt,h,w,sigma)
            target=q.gather(2,choice[...,None,None].expand(-1,-1,1,2)).squeeze(2)
            confidence=wt.gather(2,choice[...,None]).squeeze(-1)
            robust=(1-(distance/(2*sigma)).square()).clamp_min(0).square()*confidence
            theta=fit_similarity(p,target-p,robust,h,w,prior=theta)
        results.append(theta[0])
    return torch.stack(results),dict(points=points,queries=queries,weights=weights,sigma=sigma)


@torch.no_grad()
def choose_candidate(base_theta, rescue_theta, evidence, h, w, margin=.02):
    """One fixed image-evidence rule for every sample; no oracle decisions."""
    decisions=[]; stats=[]
    for b in range(len(base_theta)):
        pair=torch.stack([base_theta[b],rescue_theta[b]])
        score,n,spread,_,_=consensus(pair,evidence['points'][b:b+1],evidence['queries'][b:b+1],
                                   evidence['weights'][b:b+1],h,w,evidence['sigma'])
        enough=(n[1]>=max(12,math.ceil(evidence['points'].shape[1]*.05)))&(spread[1]>=6)
        use=enough&(score[1]>score[0]+margin)
        decisions.append(use)
        stats.append(dict(base_score=float(score[0]),rescue_score=float(score[1]),
                          rescue_inliers=int(n[1]),rescue_cells=int(spread[1]),selected_rescue=bool(use)))
    return torch.stack(decisions),stats


class GlobalRescueRefiner(nn.Module):
    def __init__(self, local=None, local_config=None, global_grid=64, hypotheses=1024, margin=.02):
        super().__init__()
        self.local=local if local is not None else GeometryRefiner(**(local_config or {}))
        self.global_matcher=GlobalMatcher(self.local,global_grid,self.local.anchors)
        self.hypotheses=hypotheses;self.margin=margin
        self.config=dict(local_config=self.local.config,global_grid=global_grid,hypotheses=hypotheses,margin=margin)

    @torch.no_grad()
    def forward(self,image0,image1,flow_init,return_aux=False,disable_rescue=False):
        if image0.shape!=image1.shape or image0.shape[-2:]!=flow_init.shape[-2:]:
            raise ValueError('Expected two same-size images and native-pixel flow')
        initial=flow_init.detach();h,w=initial.shape[-2:]
        baseline,local_aux=self.local(image0.detach(),image1.detach(),initial,return_aux=True)
        if disable_rescue:
            aux=dict(baseline=baseline,rescue=baseline,coarse=baseline,
                     selection=[dict(selected_rescue=False) for _ in range(len(initial))])
            return (baseline,aux) if return_aux else baseline
        rec=self.global_matcher(image0,image1)
        theta,evidence=robust_pose(rec,local_aux['theta'],h,w,self.hypotheses)
        coarse=dense_similarity(theta,h,w)
        rescued,rescue_aux=self.local(image0,image1,coarse,return_aux=True)
        selected,selection=choose_candidate(local_aux['theta'],rescue_aux['theta'],evidence,h,w,self.margin)
        output=torch.where(selected[:,None,None,None],rescued,baseline)
        aux=dict(baseline=baseline,rescue=rescued,coarse=coarse,selection=selection)
        return (output,aux) if return_aux else output


def perturb_initial(flow, probability=.5):
    """TRAIN ONLY: moderate errors compatible with the local capture window.

    Changes ONLY flow_init, never images or GT. Rotation <=2deg, scale <=2%,
    translation <=12 native pixels at 512. Global branch does not need this:
    it searches the whole target and is independent of initial displacement.
    """
    b,_,h,w=flow.shape
    r=torch.rand(b,5,device=flow.device,dtype=flow.dtype)
    angle=(2*r[:,0]-1)*math.radians(2)
    scale=torch.exp((2*r[:,1]-1)*.02)
    s=max(h-1,w-1)/2
    theta=torch.stack([(scale*angle.cos()-1)*s,scale*angle.sin()*s,
                       (2*r[:,2]-1)*12*w/512,(2*r[:,3]-1)*12*h/512],-1)
    noise=dense_similarity(theta,h,w)*(r[:,4]<probability)[:,None,None,None]
    return flow.detach()+noise
