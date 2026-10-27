"""D1: spatially distinct candidates only; all learned weights are unchanged.

Uses the same support threshold, score margin, random seed, hypothesis count,
scale limits, IRLS and local model as the G1+L1 plugin. Forward has no GT input.
"""
import math
import torch
from global_rescue import GlobalRescueRefiner,two_point_theta,consensus,choose_candidate
from geo_core_v1 import fit_similarity,dense_similarity


def diverse_correspondences(record):
    """Four peaks separated by a 5x5 exclusion box in the 64x64 target grid.

    Never average unrelated peaks. Keep each alternative as its own geometry
    hypothesis. Confidence and normalization match the original dual softmax.
    """
    logits=record['logits'];grid=record['grid']
    dual=(logits.log_softmax(-1)+logits.log_softmax(1)).exp()
    work=dual.clone();scores=[];indices=[]
    flat=torch.arange(grid*grid,device=logits.device)
    x=flat%grid;y=flat//grid
    for _ in range(4):
        score,index=work.max(-1)
        scores.append(score.clamp_min(0));indices.append(index)
        near=((x[None,None]-index[:,:,None]%grid).abs()<=2)&((y[None,None]-index[:,:,None]//grid).abs()<=2)
        work.masked_fill_(near,-1)
    confidence=torch.stack(scores,-1);index=torch.stack(indices,-1)
    targets=record['targets']
    queries=targets[:,None].expand(-1,index.shape[1],-1,-1).gather(2,index[...,None].expand(-1,-1,-1,2))
    weights=(confidence/confidence[:,:,0].mean(-1)[:,None,None].clamp_min(1e-8)).clamp(max=3.)
    return record['points'],queries,weights


class DiverseRescueRefiner(GlobalRescueRefiner):
    @torch.no_grad()
    def forward(self,image0,image1,flow_init,return_aux=False,disable_rescue=False):
        if disable_rescue:
            return super().forward(image0,image1,flow_init,return_aux,True)
        if image0.shape!=image1.shape or image0.shape[-2:]!=flow_init.shape[-2:]:
            raise ValueError('Expected two same-size images and native-pixel flow')
        initial=flow_init.detach();h,w=initial.shape[-2:]
        baseline,local_aux=self.local(image0.detach(),image1.detach(),initial,return_aux=True)
        record=self.global_matcher(image0,image1)
        theta,evidence=robust_pose_diverse(record,local_aux['theta'],h,w,self.hypotheses)
        coarse=dense_similarity(theta,h,w)
        rescued,rescue_aux=self.local(image0,image1,coarse,return_aux=True)
        selected,selection=choose_candidate(local_aux['theta'],rescue_aux['theta'],evidence,h,w,self.margin)
        output=torch.where(selected[:,None,None,None],rescued,baseline)
        aux=dict(baseline=baseline,rescue=rescued,coarse=coarse,selection=selection)
        return (output,aux) if return_aux else output


@torch.no_grad()
def robust_pose_diverse(record, prior, h, w, hypotheses=1024):
    points,queries,weights=diverse_correspondences(record)
    k=queries.shape[2]
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
        pp=p[0,selected//k];qq=q[0,selected//k,selected%k]
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
