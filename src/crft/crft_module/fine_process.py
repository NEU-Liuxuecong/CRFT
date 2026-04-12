import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import torch.utils.checkpoint as cp

def checkpointed(module, *args):
    return cp.checkpoint(lambda *x: module(*x), *args)


class Mlp(nn.Module):
    """Multi-Layer Perceptron (MLP)"""

    def __init__(self,
                 in_dim,
                 hidden_dim=None,
                 out_dim=None,
                 act_layer=nn.GELU):
        """
        Args:
            in_dim: input features dimension
            hidden_dim: hidden features dimension
            out_dim: output features dimension
            act_layer: activation function
        """
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.view(*x_size[:-1], self.out_dim)
        return x
    

class VanillaAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 proj_bias=False):
        super().__init__()
        """
        Args:
            dim: feature dimension
            num_heads: number of attention head
            proj_bias: bool use query, key, value bias
        """
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax_temp = self.head_dim ** -0.5
        self.kv_proj = nn.Linear(dim, dim * 2, bias=proj_bias)
        self.q_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.merge = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv=None):
        """
        Args:
            x_q (torch.Tensor): [N, L, C]
            x_kv (torch.Tensor): [N, S, C]
        """
        if x_kv is None:
            x_kv = x_q
        bs, _, dim = x_q.shape
        bs, _, dim = x_kv.shape
        # [N, S, 2, H, D] => [2, N, H, S, D]
        kv = self.kv_proj(x_kv).reshape(bs, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # [N, L, H, D] => [N, H, L, D]
        q = self.q_proj(x_q).reshape(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k, v = kv[0].transpose(-2, -1).contiguous(), kv[1].contiguous() # [N, H, D, S], [N, H, S, D]
        attn = (q @ k) * self.softmax_temp # [N, H, L, S]
        attn = attn.softmax(dim=-1)
        x_q = (attn @ v).transpose(1, 2).reshape(bs, -1, dim)
        x_q = self.merge(x_q)
        return x_q
    

class CrossBidirectionalAttention(nn.Module):
    def __init__(self, dim, num_heads, proj_bias = False):
        super().__init__()
        """
        Args:
            dim: feature dimension
            num_heads: number of attention head
            proj_bias: bool use query, key, value bias
        """

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax_temp = self.head_dim ** -0.5
        self.qk_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.v_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.merge = nn.Linear(dim, dim, bias=proj_bias)
        self.temperature = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def map_(self, func, x0, x1):
        return func(x0), func(x1)

    def forward(self, x0, x1):
        """
        Args:
            x0 (torch.Tensor): [N, L, C]
            x1 (torch.Tensor): [N, S, C]
        """
        bs = x0.size(0)

        qk0, qk1 = self.map_(self.qk_proj, x0, x1)
        v0, v1 = self.map_(self.v_proj, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.reshape(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous(),
            (qk0, qk1, v0, v1))

        qk0, qk1 = qk0 * self.softmax_temp**0.5, qk1 * self.softmax_temp**0.5
        sim = qk0 @ qk1.transpose(-2,-1).contiguous()
        attn01 = F.softmax(sim, dim=-1)
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1) 
        x0 = attn01 @ v1
        x1 = attn10 @ v0
        x0, x1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2),
                        x0, x1)
        x0, x1 = self.map_(self.merge, x0, x1)

        return x0, x1
    

class SwinPosEmbMLP(nn.Module):
    def __init__(self,
                 dim):
        super().__init__()
        self.pos_embed = None
        self.pos_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(512, dim, bias=False))
        
    def forward(self, x):
        seq_length = x.shape[1] 
        if self.pos_embed is None or self.training:
            seq_length = int(seq_length**0.5)
            coords = torch.arange(0, seq_length, device=x.device, dtype = x.dtype)
            grid = torch.stack(torch.meshgrid([coords, coords])).contiguous().unsqueeze(0)
            grid -= seq_length // 2
            grid /= (seq_length // 2)
            self.pos_embed = self.pos_mlp(grid.flatten(2).transpose(1,2))
        x = x + self.pos_embed
        return x


class WindowSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_coef, use_pre_pos_embed=False):
        super().__init__()
        self.mlp = Mlp(in_dim=dim*2, hidden_dim=dim*mlp_hidden_coef, out_dim=dim, act_layer=nn.GELU)
        self.gamma = nn.Parameter(torch.ones(dim))  
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = VanillaAttention(dim, num_heads=num_heads)
        self.pos_embed = SwinPosEmbMLP(dim)
        self.pos_embed_pre = SwinPosEmbMLP(dim) if use_pre_pos_embed else nn.Identity()

    def forward(self, x, x_pre):
        # [N, L, W^2, C] -> [N, L*W^2, C]
        N, L, WW, C = x.shape
        N_pre, L_pre, WW_pre, C_pre = x_pre.shape
        
        x = x.reshape(N, L * WW, C)
        x_pre = x_pre.reshape(N_pre, L_pre * WW_pre, C_pre)
        
        ww = x.shape[1]
        ww_pre = x_pre.shape[1]
        x = self.pos_embed(x)
        x_pre = self.pos_embed_pre(x_pre)
        x = torch.cat((x, x_pre), dim=1)
        x = x + self.gamma*self.norm1(self.mlp(torch.cat([x, self.attn(self.norm2(x))], dim=-1)))
        x, x_pre = x.split([ww, ww_pre], dim=1)
        
        # [N, L*W^2, C] -> [N, L, W^2, C]
        x = x.reshape(N, L, WW, C)
        x_pre = x_pre.reshape(N_pre, L_pre, WW_pre, C_pre)
        
        return x, x_pre


class WindowCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_coef):
        super().__init__()  
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_dim=dim*2, hidden_dim=dim*mlp_hidden_coef, out_dim=dim, act_layer=nn.GELU)
        self.cross_attn = CrossBidirectionalAttention(dim, num_heads=num_heads, proj_bias=False)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x0, x1):
        # Handle window feature dimensions: [N, L, W^2, C] -> [N, L*W^2, C]
        N0, L0, WW0, C0 = x0.shape
        N1, L1, WW1, C1 = x1.shape
        
        x0 = x0.reshape(N0, L0 * WW0, C0)
        x1 = x1.reshape(N1, L1 * WW1, C1)
        
        m_x0, m_x1 = self.cross_attn(self.norm1(x0), self.norm1(x1))
        x0 = x0 + self.gamma*self.norm2(self.mlp(torch.cat([x0, m_x0], dim=-1)))
        x1 = x1 + self.gamma*self.norm2(self.mlp(torch.cat([x1, m_x1], dim=-1)))
        
        # Restore window feature dimensions: [N, L*W^2, C] -> [N, L, W^2, C]
        x0 = x0.reshape(N0, L0, WW0, C0)
        x1 = x1.reshape(N1, L1, WW1, C1)
        
        return x0, x1


class FineProcess(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        block_dims = config['resnet']['block_dims']
        self.block_dims = block_dims
        self.W_f = config['fine_window_size']
        self.W_m = config['medium_window_size'] 
        nhead_f = config["fine"]['nhead_fine_level']
        nhead_m = config["fine"]['nhead_medium_level']  
        mlp_hidden_coef = config["fine"]['mlp_hidden_dim_coef'] 

        # Networks
        self.conv_merge = nn.Sequential(nn.Conv2d(block_dims[2]*2, block_dims[1], kernel_size=1, stride=1, padding=0, bias=False),
                                        nn.Conv2d(block_dims[1], block_dims[1], kernel_size=3, stride=1, padding=1, groups=block_dims[1], bias=False),
                                        nn.BatchNorm2d(block_dims[1])
                                        )
        self.out_conv_m = nn.Conv2d(block_dims[1], block_dims[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.out_conv_f = nn.Conv2d(block_dims[0], block_dims[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.self_attn_m = WindowSelfAttention(block_dims[1], num_heads=nhead_m,
                                                mlp_hidden_coef=mlp_hidden_coef, use_pre_pos_embed=False)
        self.cross_attn_m = WindowCrossAttention(block_dims[1], num_heads=nhead_m,
                                                  mlp_hidden_coef=mlp_hidden_coef)
        self.self_attn_f = WindowSelfAttention(block_dims[0], num_heads=nhead_f,
                                                mlp_hidden_coef=mlp_hidden_coef, use_pre_pos_embed=True)
        self.cross_attn_f = WindowCrossAttention(block_dims[0], num_heads=nhead_f,
                                                  mlp_hidden_coef=mlp_hidden_coef)
        self.down_proj_m_f = nn.Linear(block_dims[1], block_dims[0], bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def pre_process(self, feat_f0, feat_f1, feat_m0, feat_m1, feat_c0, feat_c1, feat_c0_pre, feat_c1_pre, data):
        W_f = self.W_f
        W_m = self.W_m
        data.update({'W_f': W_f, 'W_m': W_m})

        feat_c0 = rearrange(feat_c0, 'n (h w) c -> n c h w', h=data["hw0_c"][0], w=data["hw0_c"][1])
        feat_c1 = rearrange(feat_c1, 'n (h w) c -> n c h w', h=data["hw1_c"][0], w=data["hw1_c"][1])
        feat_c0 = self.conv_merge(torch.cat([feat_c0, feat_c0_pre], dim=1))
        feat_c1 = self.conv_merge(torch.cat([feat_c1, feat_c1_pre], dim=1))

        if feat_m0.shape[2] == feat_m1.shape[2] and feat_m0.shape[3] == feat_m1.shape[3]:
            feat_m = self.out_conv_m(torch.cat([feat_m0, feat_m1], dim=0))
            feat_m0, feat_m1 = torch.chunk(feat_m, 2, dim=0)
            feat_f = self.out_conv_f(torch.cat([feat_f0, feat_f1], dim=0))
            feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        else:
            feat_m0 = self.out_conv_m(feat_m0)
            feat_m1 = self.out_conv_m(feat_m1)
            feat_f0 = self.out_conv_f(feat_f0)
            feat_f1 = self.out_conv_f(feat_f1)

        stride_f = max(1, data['hw0_f'][0] // data['hw0_c'][0])  # ratio of fine to coarse
        stride_m = max(1, data['hw0_m'][0] // data['hw0_c'][0])  # ratio of medium to coarse
        
        if W_m == 1:
            feat_m0_unfold = rearrange(feat_m0, 'n c h w -> n (h w) 1 c')
            feat_m1_unfold = rearrange(feat_m1, 'n c h w -> n (h w) 1 c')
        else:
            feat_m0_unfold = F.unfold(feat_m0, kernel_size=(W_m, W_m), stride=stride_m, padding=W_m//2)
            feat_m0_unfold = rearrange(feat_m0_unfold, 'n (c ww) l -> n l ww c', ww=W_m**2)
            feat_m1_unfold = F.unfold(feat_m1, kernel_size=(W_m, W_m), stride=stride_m, padding=W_m//2)
            feat_m1_unfold = rearrange(feat_m1_unfold, 'n (c ww) l -> n l ww c', ww=W_m**2)

        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W_f**2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W_f**2)

        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) 1 c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) 1 c')

        return feat_c0, feat_c1, feat_m0_unfold, feat_m1_unfold, feat_f0_unfold, feat_f1_unfold

    def forward(self, feat_f0, feat_f1, feat_m0, feat_m1, feat_c0, feat_c1, feat_c0_pre, feat_c1_pre, data):
        """
        Args:
            feat_f0, feat_f1: Fine level features [N, C, H_f, W_f] (1/2 resolution)
            feat_m0, feat_m1: Medium level features [N, C, H_m, W_m] (1/4 resolution) 
            feat_c0, feat_c1: Coarse level features [N, L, C] (1/8 resolution)
            feat_c0_pre, feat_c1_pre: Raw coarse features [N, C, H_c, W_c]
            data: Dictionary containing size info
            
        Returns:
            feat_f0_unfold, feat_f1_unfold: Enhanced dense fine features [N, L_f, W_f^2, C]
        """
        
        feat_c0, feat_c1, feat_m0_unfold, feat_m1_unfold, \
            feat_f0_unfold, feat_f1_unfold = self.pre_process(feat_f0, feat_f1, feat_m0, feat_m1,
                                                               feat_c0, feat_c1, feat_c0_pre, feat_c1_pre, data)
        
        # 1. Self attention (c + m) 
        feat_m_unfold, _ = self.self_attn_m(torch.cat([feat_m0_unfold, feat_m1_unfold], dim=0),
                                           torch.cat([feat_c0, feat_c1], dim=0))
        feat_m0_unfold, feat_m1_unfold = torch.chunk(feat_m_unfold, 2, dim=0)

        # 2. Cross attention 
        feat_m0_unfold, feat_m1_unfold = self.cross_attn_m(feat_m0_unfold, feat_m1_unfold)

        # 3. medium-fine
        feat_m_unfold = self.down_proj_m_f(torch.cat([feat_m0_unfold, feat_m1_unfold], dim=0))
        feat_m0_unfold, feat_m1_unfold = torch.chunk(feat_m_unfold, 2, dim=0)

        # 4. Self attention (m + f) 
        feat_f_unfold, _ = self.self_attn_f(torch.cat([feat_f0_unfold, feat_f1_unfold], dim=0),
                                           torch.cat([feat_m0_unfold, feat_m1_unfold], dim=0))
        feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_f_unfold, 2, dim=0)

        # 5. Cross attention 
        feat_f0_unfold, feat_f1_unfold = self.cross_attn_f(feat_f0_unfold, feat_f1_unfold)
       
        return feat_f0_unfold, feat_f1_unfold