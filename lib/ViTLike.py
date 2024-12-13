

from timm.layers import trunc_normal_
import torch.nn as nn
import torch
from timm.layers import DropPath

DROPOUT_FLOPS = 4
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8
SOFTMAX_FLOPS = 5


def get_dpr(drop_path_rate, depth, drop_path_decay='linear'):
    if drop_path_decay == 'linear':
        # linear dpr decay
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    elif drop_path_decay == 'fix':
        # use fixed dpr
        dpr = [drop_path_rate] * depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate) == depth
        dpr = drop_path_rate
    return dpr


class GroupLinear(nn.Module):
    '''
    Group Linear operator 
    '''

    def __init__(self, in_planes, out_channels, groups=1, bias=True):
        super(GroupLinear, self).__init__()
        assert in_planes % groups == 0
        assert out_channels % groups == 0
        self.in_dim = in_planes
        self.out_dim = out_channels
        self.groups = groups
        self.bias = bias
        self.group_in_dim = int(self.in_dim / self.groups)
        self.group_out_dim = int(self.out_dim / self.groups)

        self.group_weight = nn.Parameter(torch.zeros(
            self.groups, self.group_in_dim, self.group_out_dim))
        self.group_bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x):
        t, b, d = x.size()
        x = x.view(t, b, self.groups, int(d / self.groups))
        out = torch.einsum('tbgd,gdf->tbgf', (x, self.group_weight)
                           ).reshape(t, b, self.out_dim) + self.group_bias
        return out

    def extra_repr(self):
        s = ('{in_dim}, {out_dim}')
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Mlp(nn.Module):
    '''
    MLP with support to use group linear operator
    FC + ACT + DROP + FC + DROP
    '''

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., group=1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group == 1:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = GroupLinear(in_features, hidden_features, group)
            self.fc2 = GroupLinear(hidden_features, out_features, group)
        self.act = act_layer()

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    '''
    Multi-head self-attention
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with some modification to support different num_heads and head_dim.

    [B, N, C] -> [B, N, dim]
    '''

    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, self.head_dim *
                             self.num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, padding_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        # B,heads,N,C/heads
        q, k, v = qkv[0], qkv[1], qkv[2]

        # trick here to make q@k.t more stable
        attn = ((q * self.scale) @ k.transpose(-2, -1))  # [B, heads, N, N]
        if padding_mask is not None:
            attn = attn.view(B, self.num_heads, N, N)
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_float = attn.softmax(dim=-1, dtype=torch.float32)
            attn = attn_float.type_as(attn)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N,
                                               # [B, N, d]
                                               self.head_dim * self.num_heads)
        x = self.proj(x)  # [B, N, dim]
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    '''
    Pre-layernorm transformer block
    '''

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, skip_lam=1.):
        super().__init__()
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.skip_lam = skip_lam

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            # [B, N, C] -> [B, N, dim]
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, group=group)

    def forward(self, x, padding_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x),
                               padding_mask)) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x

    def flops(self, s):
        heads = self.attn.num_heads
        h = self.dim
        i = self.mlp_hidden_dim
        mha_block_flops = dict(
            kqv=3 * h * h,
            attention_scores=h * s,
            attn_softmax=SOFTMAX_FLOPS * s * heads,
            attention_dropout=DROPOUT_FLOPS * s * heads,
            attention_scale=s * heads,
            attention_weighted_avg_values=h * s,
            attn_output=h * h,
            attn_output_bias=h,
            attn_output_dropout=DROPOUT_FLOPS * h,
            attn_output_residual=h,
            attn_output_layer_norm=LAYER_NORM_FLOPS * h,)
        ffn_block_flops = dict(
            intermediate=h * i,
            intermediate_act=ACTIVATION_FLOPS * i,
            intermediate_bias=i,
            output=h * i,
            output_bias=h,
            output_dropout=DROPOUT_FLOPS * h,
            output_residual=h,
            output_layer_norm=LAYER_NORM_FLOPS * h,)

        return sum(mha_block_flops.values()) * s + sum(ffn_block_flops.values()) * s


class ViTLike(nn.Module):
    """ ViTLike with Vision Transformer
    Arguements:
        num_classes: The slide-level class nummbers (default: 4)
        embed_dim: The instance feature dimension (default: 1280)
        depth: The numbers of Transformer blocks (default: 2)
        num_heads: The numbers of Transformer block head (default: 12)
        skip_lam: residual scalar for skip connection (default: 1.0)
    """

    def __init__(self, num_classes=4, embed_dim=1280, depth=2,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', norm_layer=nn.LayerNorm, head_dim=None,
                 skip_lam=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.output_dim = num_classes
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # [0.0, 0.0066666668, 0.0133333336, ..., 0.1]
        dpr = get_dpr(drop_path_rate, depth, drop_path_decay)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.slide_head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # init weight
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # [B, num_patches + 1， embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx x.shape', x.shape)

        # token interaction
        x = self.forward_tokens(x)  # [B, num_patches + 1, embed_dim]

        # slide-level prediction
        x_cls = self.slide_head(x[:, 0])  # [B, num_classes]
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx x_cls', x_cls.shape)
        return x_cls
