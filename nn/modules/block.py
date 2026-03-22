# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from einops import rearrange

from ..modules.conv import Conv, DWConv
# from ..modules.block import C2f, Bottleneck

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'ConvNormLayer', 'BasicBlock', 
           'BottleNeck', 'Blocks','CSFH_Block', 'MANet','SFHF_Block','MANet_C2fHFERB','MHE','MPDF','SFR')


import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmcv.ops import ModulatedDeformConv2dPack
except Exception:
    ModulatedDeformConv2dPack = None


class DWConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, tuple):
                p = tuple(kk // 2 for kk in k)
            else:
                p = k // 2
        self.dw = nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=p, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DeformableBranch(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        if ModulatedDeformConv2dPack is not None:
            self.op = nn.Sequential(
                ModulatedDeformConv2dPack(c1, c2, kernel_size=k, stride=1, padding=k // 2, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )
        else:
            self.op = DWConvBNAct(c1, c2, k=k, s=1, p=k // 2)

    def forward(self, x):
        return self.op(x)


class BranchAttention(nn.Module):
    def __init__(self, c1, num_branches=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c1, num_branches, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        alpha = self.fc(self.pool(x)).flatten(1)
        alpha = F.softmax(alpha, dim=1)
        return alpha


class MSDConv(nn.Module):
    def __init__(self, c1, c2, k=3, M=9):
        super().__init__()
        assert c2 % 4 == 0, "c2 must be divisible by 4"
        c_branch = c2 // 4

        self.branch1 = DWConvBNAct(c1, c_branch, k=(k, k), s=1)
        self.branch2 = DWConvBNAct(c1, c_branch, k=(1, M), s=1)
        self.branch3 = DWConvBNAct(c1, c_branch, k=(M, 1), s=1)
        self.branch4 = DeformableBranch(c1, c_branch, k=3)

        self.attn = BranchAttention(c1, num_branches=4)

    def forward(self, x):
        alpha = self.attn(x)

        y1 = self.branch1(x) * alpha[:, 0].view(-1, 1, 1, 1)
        y2 = self.branch2(x) * alpha[:, 1].view(-1, 1, 1, 1)
        y3 = self.branch3(x) * alpha[:, 2].view(-1, 1, 1, 1)
        y4 = self.branch4(x) * alpha[:, 3].view(-1, 1, 1, 1)

        out = torch.cat([y1, y2, y3, y4], dim=1)
        return out


class DirectionalScaleEstimation(nn.Module):
    def __init__(self, channels, max_kernel=31):
        super().__init__()
        max_kernel = max(3, max_kernel)
        if max_kernel % 2 == 0:
            max_kernel -= 1

        self.channels = channels
        self.max_kernel = max_kernel

        self.weight_h = nn.Parameter(torch.randn(channels, 1, max_kernel) * 0.02)
        self.weight_w = nn.Parameter(torch.randn(channels, 1, max_kernel) * 0.02)
        self.bias_h = nn.Parameter(torch.zeros(channels))
        self.bias_w = nn.Parameter(torch.zeros(channels))

    def _odd_kernel(self, n):
        k = min(self.max_kernel, n if n % 2 == 1 else n - 1)
        k = max(1, k)
        if k % 2 == 0:
            k -= 1
        return k

    def _slice_weight(self, weight, k):
        start = (self.max_kernel - k) // 2
        end = start + k
        return weight[:, :, start:end].contiguous()

    def forward(self, x):
        b, c, h, w = x.shape

        h_ctx = x.mean(dim=3)
        w_ctx = x.mean(dim=2)

        kh = self._odd_kernel(h)
        kw = self._odd_kernel(w)

        wh = self._slice_weight(self.weight_h, kh)
        ww = self._slice_weight(self.weight_w, kw)

        y_h = F.conv1d(h_ctx, wh, bias=self.bias_h, padding=kh // 2, groups=c)
        y_w = F.conv1d(w_ctx, ww, bias=self.bias_w, padding=kw // 2, groups=c)

        y_h = y_h.unsqueeze(-1).expand(-1, -1, -1, w)
        y_w = y_w.unsqueeze(-2).expand(-1, -1, h, -1)

        return y_h, y_w


class CEU(nn.Module):
    def __init__(self, channels, max_kernel=31):
        super().__init__()
        self.dse = DirectionalScaleEstimation(channels, max_kernel=max_kernel)
        self.fuse = ConvBNAct(channels * 2, channels, k=1, s=1, p=0)

    def forward(self, x):
        y_h, y_w = self.dse(x)
        out = torch.cat([y_h, y_w], dim=1)
        out = self.fuse(out)
        return out


class GLUController(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels * 2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        return F.glu(self.proj(x), dim=1)


class CEUBlock(nn.Module):
    def __init__(self, channels, max_kernel=31):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.ceu = CEU(channels, max_kernel=max_kernel)
        self.gamma1 = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.bn2 = nn.BatchNorm2d(channels)
        self.glu = GLUController(channels)
        self.gamma2 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = x + self.gamma1 * self.ceu(self.bn1(x))
        y = y + self.gamma2 * self.glu(self.bn2(y))
        return y


class MPDF(nn.Module):
    def __init__(self, c1, c2, k=3, M=9, max_kernel=31):
        super().__init__()
        self.proj_in = ConvBNAct(c1, c2, k=1, s=1, p=0)
        self.msd_conv = MSDConv(c2, c2, k=k, M=M)
        self.ceu_block = CEUBlock(c2, max_kernel=max_kernel)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.msd_conv(x)
        x = self.ceu_block(x)
        return x


class HGBlock(nn.Module):
  
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

################################### RT-DETR PResnet ###################################
def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', input_resolution=None, sr_ratio=None, kernel_size=None, kan_name=None, variant='d'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            if input_resolution is not None and sr_ratio is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        input_resolution=input_resolution,
                        sr_ratio=sr_ratio)
                )
            elif kernel_size is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kernel_size=kernel_size)
                )
            elif kan_name is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kan_name=kan_name)
                )
            else:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act)
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class SFHF_FFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(SFHF_FFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 2, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)

        return x


class TokenMixer_For_Local(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(TokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2

        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.cat([cd1, cd2], dim=1)

        return x


class SFHF_FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=4):
        # bn_layer not used
        super(SFHF_FourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)

        self.fdc = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2 * self.groups,
                             kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True)

        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                             padding=1, stride=1, groups=in_channels * 2, bias=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, self.groups, 2 * c, h, -1)  # (batch, c*2, h, w/2+1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class TokenMixer_For_Gloal(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(TokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU()
        )
        self.FFC = SFHF_FourierUnit(self.dim * 2, self.dim * 2)

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x + x0)

        return x


class SFHF_Mixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_local=TokenMixer_For_Local,
            token_mixer_for_gloal=TokenMixer_For_Gloal,
    ):
        super(SFHF_Mixer, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim, )
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, )

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * dim, 2 * dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim // 2, 2 * dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)

        return x


# class SFHF_Block(nn.Module):
#     def __init__(
#             self,
#             dim,
#             dim_out,
#             norm_layer=nn.BatchNorm2d,
#             token_mixer=SFHF_Mixer,
#     ):
#         super(SFHF_Block, self).__init__()
#         self.dim = dim
#         self.norm1 = norm_layer(dim)
#         self.norm2 = norm_layer(dim)
#         self.mixer = token_mixer(dim=self.dim)
#         self.ffn = SFHF_FFN(dim=self.dim)

#         self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

#     def forward(self, x):
#         copy = x
#         x = self.norm1(x)
#         x = self.mixer(x)
#         x = x * self.beta + copy

#         copy = x
#         x = self.norm2(x)
#         x = self.ffn(x)
#         x = x * self.gamma + copy

#         return x
class SFHF_Block(nn.Module):
    def __init__(
            self,
            dim,
            # dim_out,
            # norm_layer=nn.BatchNorm2d,
            # token_mixer=SFHF_Mixer,
    ):
        super(SFHF_Block, self).__init__()
        self.dim = dim
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = SFHF_Mixer(dim=self.dim)
        self.ffn = SFHF_FFN(dim=self.dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy

        return x


class CSFH_Block(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SFHF_Block(self.c) for _ in range(n))


class MANet(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), DWConv(dim_hid, dim_hid, kernel_size, 1),
                                        Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv_final(torch.cat(y, 1))


class HFERB(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.mid_dim = dim // 2
        self.dim = dim
        self.act = nn.GELU()
        self.last_fc = nn.Conv2d(self.dim, self.dim, 1)

        # High-frequency enhancement branch
        self.fc = nn.Conv2d(self.mid_dim, self.mid_dim, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)

        # Local feature extraction branch
        self.conv = nn.Conv2d(self.mid_dim, self.mid_dim, 3, 1, 1)

    def forward(self, x):
        self.h, self.w = x.shape[2:]
        short = x

        # Local feature extraction branch
        lfe = self.act(self.conv(x[:, :self.mid_dim, :, :]))

        # High-frequency enhancement branch
        hfe = self.act(self.fc(self.max_pool(x[:, self.mid_dim:, :, :])))

        x = torch.cat([lfe, hfe], dim=1)
        x = short + self.last_fc(x)
        return x


class C2f_HFERB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(HFERB(self.c) for _ in range(n))


class MANet_C2fHFERB(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(C2f_HFERB(self.c, self.c) for _ in range(n))


class MHE(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(HFERB(self.c) for _ in range(n))


class LayerNorm(nn.Module):
    def __init__(self, dim, norm_type='WithBias', eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1)) if norm_type == 'WithBias' else None
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            return x_norm * self.weight + self.bias
        else:
            return x_norm * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.project_out(x)
        return x


class GSA(nn.Module):
    def __init__(self, channels, num_heads=8, bias=False):
        super(GSA, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        if prev_atns is None:
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = self.act(attn)
            out = attn @ v
            y = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
            y = self.project_out(y)
            return y, attn
        else:
            attn = prev_atns
            v = rearrange(x, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            out = attn @ v
            y = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
            y = self.project_out(y)
            return y


class RSA(nn.Module):
    def __init__(self, channels, num_heads, shifts=1, window_sizes=4, bias=False):
        super(RSA, self).__init__()
        self.channels = channels
        self.shifts = shifts
        if isinstance(window_sizes, (list, tuple)):
            self.window_size = int(window_sizes[0])
        else:
            self.window_size = int(window_sizes)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        wsize = self.window_size
        if prev_atns is None:
            x_ = x
            if self.shifts > 0:
                x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
            qkv = self.qkv_dwconv(self.qkv(x_))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)
            k = rearrange(k, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)
            v = rearrange(v, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q.transpose(-2, -1) @ k) * self.temperature
            attn = self.act(attn)
            out = v @ attn
            out = rearrange(
                out,
                "b (h w) (dh dw) c -> b c (h dh) (w dw)",
                h=h // wsize,
                w=w // wsize,
                dh=wsize,
                dw=wsize,
            )
            if self.shifts > 0:
                out = torch.roll(out, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
            y = self.project_out(out)
            return y, attn
        else:
            if self.shifts > 0:
                x = torch.roll(x, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
            atn = prev_atns
            v = rearrange(
                x,
                "b c (h dh) (w dw) -> b (h w) (dh dw) c",
                dh=wsize,
                dw=wsize,
            )
            y_ = v @ atn
            y_ = rearrange(
                y_,
                "b (h w) (dh dw) c -> b c (h dh) (w dw)",
                h=h // wsize,
                w=w // wsize,
                dh=wsize,
                dw=wsize,
            )
            if self.shifts > 0:
                y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
            y = self.project_out(y_)
            return y


class FDT(nn.Module):
    def __init__(self, inp_channels, num_heads=4, window_sizes=4, shifts=0, shared_depth=1, ffn_expansion_factor=2.66):
        super(FDT, self).__init__()
        self.shared_depth = shared_depth

        modules_ffd = {}
        modules_att = {}
        modules_norm = {}
        for i in range(shared_depth):
            modules_ffd[f"ffd{i}"] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modules_att[f"att_{i}"] = RSA(
                channels=inp_channels,
                num_heads=num_heads,
                shifts=shifts,
                window_sizes=window_sizes,
            )
            modules_norm[f"norm_{i}"] = LayerNorm(inp_channels, "WithBias")
            modules_norm[f"norm_{i+2}"] = LayerNorm(inp_channels, "WithBias")
        self.modules_ffd = nn.ModuleDict(modules_ffd)
        self.modules_att = nn.ModuleDict(modules_att)
        self.modules_norm = nn.ModuleDict(modules_norm)

        modulec_ffd = {}
        modulec_att = {}
        modulec_norm = {}
        for i in range(shared_depth):
            modulec_ffd[f"ffd{i}"] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modulec_att[f"att_{i}"] = GSA(channels=inp_channels, num_heads=num_heads)
            modulec_norm[f"norm_{i}"] = LayerNorm(inp_channels, "WithBias")
            modulec_norm[f"norm_{i+2}"] = LayerNorm(inp_channels, "WithBias")
        self.modulec_ffd = nn.ModuleDict(modulec_ffd)
        self.modulec_att = nn.ModuleDict(modulec_att)
        self.modulec_norm = nn.ModuleDict(modulec_norm)

    def forward(self, x):
        atn = None
        B, C, H, W = x.size()
        for i in range(self.shared_depth):
            if i == 0:
                x_, atn = self.modules_att[f"att_{i}"](self.modules_norm[f"norm_{i}"](x), None)
                x = self.modules_ffd[f"ffd{i}"](self.modules_norm[f"norm_{i+2}"](x_ + x)) + x_
            else:
                x_ = self.modules_att[f"att_{i}"](self.modules_norm[f"norm_{i}"](x), atn)
                x = self.modules_ffd[f"ffd{i}"](self.modules_norm[f"norm_{i+2}"](x_ + x)) + x_

        for i in range(self.shared_depth):
            if i == 0:
                x_, atn = self.modulec_att[f"att_{i}"](self.modulec_norm[f"norm_{i}"](x), None)
                x = self.modulec_ffd[f"ffd{i}"](self.modulec_norm[f"norm_{i+2}"](x_ + x)) + x_
            else:
                x_ = self.modulec_att[f"att_{i}"](self.modulec_norm[f"norm_{i}"](x), atn)
                x = self.modulec_ffd[f"ffd{i}"](self.modulec_norm[f"norm_{i+2}"](x_ + x)) + x_
        return x


class C2f_FDT(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FDT(self.c) for _ in range(n))


class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        haar_weights = torch.ones(4, 1, 2, 2)
        haar_weights[1, 0, 0, 1] = -1
        haar_weights[1, 0, 1, 1] = -1
        haar_weights[2, 0, 1, 0] = -1
        haar_weights[2, 0, 1, 1] = -1
        haar_weights[3, 0, 1, 0] = -1
        haar_weights[3, 0, 0, 1] = -1

        haar_weights = torch.cat([haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(haar_weights, requires_grad=grad)

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape(x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2)
            out = torch.transpose(out, 1, 2)
            out = out.reshape(x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2)
            return out
        else:
            out = x.reshape(x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3])
            out = torch.transpose(out, 1, 2)
            out = out.reshape(x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels)



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SFR(nn.Module):
    def __init__(self, chn, ph=2, pl=4, token_expansion=2.0):
        super().__init__()
        dim_big, dim_small = chn
        self.dim_big = dim_big
        self.dim_small = dim_small
        self.ph = ph
        self.pl = pl

        self.HaarWavelet = HaarWavelet(dim_big, grad=False)
        self.InverseHaarWavelet = HaarWavelet(dim_big, grad=False)

        token_dim = dim_big
        hidden = max(1, int(token_dim * token_expansion))

        self.hf_mlp1 = nn.Linear(token_dim, hidden)
        self.hf_norm = nn.LayerNorm(hidden)
        self.hf_mlp2 = nn.Linear(hidden, token_dim)
        self.hf_prompt = nn.Parameter(torch.randn(1, 1, token_dim))

        self.low_proj = (
            nn.Conv2d(dim_small, dim_big, kernel_size=1, bias=True)
            if dim_small is not None and dim_small > 0
            else None
        )

        self.lf_gate1 = nn.Linear(dim_big, hidden)
        self.lf_gate_norm = nn.LayerNorm(hidden)
        self.lf_gate2 = nn.Linear(hidden, dim_big)

    def _enhance_high_band(self, band):
        b, c, h, w = band.shape
        ph = self.ph

        if h % ph != 0 or w % ph != 0:
            return band

        patches = rearrange(
            band,
            "b c (hh ph) (ww pw) -> b (hh ww) c ph pw",
            ph=ph,
            pw=ph,
        )

        z = patches.abs().mean(dim=(-1, -2))

        q = self.hf_mlp1(z)
        q = self.hf_norm(q)
        q = self.hf_mlp2(q)

        prompt = self.hf_prompt.expand(b, q.size(1), -1)
        sim = F.cosine_similarity(q, prompt, dim=-1)
        alpha = F.relu(sim).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        patches = alpha * patches

        band_enh = rearrange(
            patches,
            "b (hh ww) c ph pw -> b c (hh ph) (ww pw)",
            hh=h // ph,
            ww=w // ph,
            ph=ph,
            pw=ph,
        )
        return band_enh

    def _enhance_low_band(self, low_band, small_feat):
        if small_feat is None or self.low_proj is None:
            return low_band

        guide = self.low_proj(small_feat)

        if guide.shape[2:] != low_band.shape[2:]:
            guide = F.interpolate(
                guide,
                size=low_band.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        b, c, h, w = low_band.shape
        pl = self.pl

        if h % pl != 0 or w % pl != 0:
            k_global = guide.mean(dim=(-1, -2))
            gate = self.lf_gate1(k_global)
            gate = self.lf_gate_norm(gate)
            gate = self.lf_gate2(gate)
            gate = torch.sigmoid(gate).unsqueeze(-1).unsqueeze(-1)
            return gate * low_band

        v_patches = rearrange(
            low_band,
            "b c (hh ph) (ww pw) -> b (hh ww) c ph pw",
            ph=pl,
            pw=pl,
        )

        k_patches = rearrange(
            guide,
            "b c (hh ph) (ww pw) -> b (hh ww) c ph pw",
            ph=pl,
            pw=pl,
        )

        k_token = k_patches.mean(dim=(-1, -2))

        gate = self.lf_gate1(k_token)
        gate = self.lf_gate_norm(gate)
        gate = self.lf_gate2(gate)
        gate = torch.sigmoid(gate).unsqueeze(-1).unsqueeze(-1)

        v_patches = gate * v_patches

        low_enh = rearrange(
            v_patches,
            "b (hh ww) c ph pw -> b c (hh ph) (ww pw)",
            hh=h // pl,
            ww=w // pl,
            ph=pl,
            pw=pl,
        )
        return low_enh

    def forward(self, x):
        x_big, x_small = x

        haar = self.HaarWavelet(x_big, rev=False)

        a = haar.narrow(1, 0, self.dim_big)
        h = haar.narrow(1, self.dim_big, self.dim_big)
        v = haar.narrow(1, self.dim_big * 2, self.dim_big)
        d = haar.narrow(1, self.dim_big * 3, self.dim_big)

        h_enh = self._enhance_high_band(h)
        v_enh = self._enhance_high_band(v)
        d_enh = self._enhance_high_band(d)
        a_enh = self._enhance_low_band(a, x_small)

        bands = torch.cat([a_enh, h_enh, v_enh, d_enh], dim=1)
        out = self.InverseHaarWavelet(bands, rev=True)
        return out
