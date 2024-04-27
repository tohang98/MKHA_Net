import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

train_size = (1,3,256,256)
# --------------------------------------------------------------------------------

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()        
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),            
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        sattn = self.pa(x)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),            
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn
    
class ChannelShuffe(nn.Module):
    def __init__(self, dim):
        super(ChannelShuffe, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)       
        return pattn2

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MSLKC(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels        
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim),
                                  nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim),
                                  nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim),
                                  nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
                                  ])
        
        
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        
        
        self.act = nn.GELU() 
        self.norm1 = LayerNorm(dim)
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        
        self.ldk = nn.Sequential(
            
            nn.Identity(),
            nn.Conv2d(chunk_dim, chunk_dim, kernel_size=3, padding=3, groups=chunk_dim, dilation=3, padding_mode='reflect'),
            nn.Conv2d(chunk_dim, chunk_dim, kernel_size=5, padding=6, groups=chunk_dim, dilation=3, padding_mode='reflect'),
            nn.Conv2d(chunk_dim, chunk_dim, kernel_size=7, padding=9, groups=chunk_dim, dilation=3, padding_mode='reflect')      
           
        )
    def forward(self, x):
        identity0 = x
        x = self.norm1(x)
        identity = x
        h, w = x.size()[-2:]      
        
        x = self.conv1(x)
        x = self.conv2(x)
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:                
                p_size = (h//2**i, w//2**i)
                s = self.ldk[i](xc[i])
                s = F.adaptive_max_pool2d(s, p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            elif i==0:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * identity
        return out + identity0

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mode):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)        

        self.mslkc = MSLKC(in_channel)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 4, 1),
            nn.GELU(),            
            nn.Conv2d(in_channel * 4, in_channel, 1)
        )

        self.pa = PixelAttention(dim=in_channel)
        self.ca = ChannelAttention(dim=in_channel)
        self.cs = ChannelShuffe(dim=in_channel)

        self.norm = LayerNorm(in_channel)

    def forward(self, x):
        out = self.conv1(x)

        out = self.mslkc(out)
        
        
        res0 = out
        res = self.norm(res0)        
        ca = self.ca(res)
        pa = self.pa(res)
        patt1 = ca + pa       
        
        patt = self.cs(res,patt1)
        out = self.mlp(patt)
        out = out * res
        out = out + res0
        
        out = self.conv2(out)
        return out + x
