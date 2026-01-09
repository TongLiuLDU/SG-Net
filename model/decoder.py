import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from timm.models.layers import DropPath, to_2tuple
import torchvision



def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def get_bn(dim, use_sync_bn=False):
    return nn.BatchNorm2d(dim)

class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.ratio = ratio
        activation='relu'
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False)
        
        if(activation=='leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif(activation=='gelu'):
            self.activation = nn.GELU()
        elif(activation=='relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif(activation=='hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:    
            self.activation = nn.ReLU(inplace=True)
            
        self.fc2   = nn.Conv2d(in_planes // self.ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= self.max_pool(x)

        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class GatedMLP(nn.Module):
    """ Gated MLP block for CustomContMix """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features // 2, hidden_features // 2, 3, padding=1, groups=hidden_features // 2)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features // 2, out_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x, v = x.chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.fc2(x)
        return x
    
class DilatedReparamBlock(nn.Module):
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl
        if kernel_size >= 11: self.kernel_sizes, self.dilates = [5, 7, 5, 3, 3, 3], [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9: self.kernel_sizes, self.dilates = [5, 7, 5, 3, 3], [1, 1, 2, 3, 4]
        elif kernel_size == 7: self.kernel_sizes, self.dilates = [5, 3, 3, 3], [1, 1, 2, 3]
        elif kernel_size == 5: self.kernel_sizes, self.dilates = [3, 3], [1, 2]
        else: raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(f'dil_conv_k{k}_{r}', nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1, padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels, bias=False))
                self.__setattr__(f'dil_bn_k{k}_{r}', get_bn(channels, use_sync_bn=use_sync_bn))
    def forward(self, x):
        if not hasattr(self, 'origin_bn'): return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv, bn = self.__getattr__(f'dil_conv_k{k}_{r}'), self.__getattr__(f'dil_bn_k{k}_{r}')
            out = out + bn(conv(x))
        return out




class CGSC(nn.Module):
    def __init__(self, channels, kernel_size=7, mlp_ratio=4, drop_path=0, deploy=False):
        super().__init__()
        self.kernel_size = kernel_size
        mlp_dim = int(channels * mlp_ratio)

        self.query_proj = nn.Conv2d(channels, channels, 1)
        self.key_proj = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.Conv2d(channels, channels, 1))
        self.value_proj = nn.Conv2d(channels, channels, 1)

        self.weight_proj = nn.Conv2d(49, kernel_size**2, kernel_size=1)

        self.norm1 = LayerNorm2d(channels)
        self.lepe = nn.Sequential(
            DilatedReparamBlock(channels, kernel_size=kernel_size, deploy=deploy),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU())
        self.fusion_proj = nn.Conv2d(channels, channels, 1)
        
        self.channel_attention = ChannelAttention(channels)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ls1 = LayerScale(channels, init_value=1e-5)
        self.ls2 = LayerScale(channels, init_value=1e-5)
        
        # Learnable residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1))

        self.norm2 = LayerNorm2d(channels)
        self.mlp = GatedMLP(in_features=channels, hidden_features=mlp_dim)

    def forward(self, x_up, x_skip):
        identity = x_skip
        
        lepe = self.lepe(x_skip)
        gate = self.gate(x_skip)


        query = self.query_proj(x_skip)
        key = self.key_proj(x_up)
        value = self.value_proj(x_skip)

        B, C, H, W = query.shape
        scale = C ** -0.5
        query_r = rearrange(query, 'b c h w -> b (h w) c')
        key_r = rearrange(key, 'b c h w -> b c (h w)')
        
        attn = torch.bmm(query_r, key_r) * scale
        attn = torch.softmax(attn, dim=-1)

        attn_r = rearrange(attn, 'b (h w) l -> b l h w', h=H, w=W)
        dynamic_weights = self.weight_proj(attn_r)

        unfolded_value = F.unfold(value, kernel_size=self.kernel_size, padding=self.kernel_size//2)
        unfolded_value_r = rearrange(unfolded_value, 'b (c k) (h w) -> b c k h w', c=C, k=self.kernel_size**2, h=H, w=W)
        mixed_features = einsum(dynamic_weights, unfolded_value_r, 'b k h w, b c k h w -> b c h w')

        x_fused = self.fusion_proj(mixed_features) + lepe
        x_fused = self.channel_attention(x_fused) * x_fused
        x_fused = gate * x_fused + identity * self.residual_scale
        return x_fused




class DeformConv(nn.Module):
    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        out = self.pointwise_conv(out)
        return out






class MultiScaleDeformConv_3x3(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleDeformConv_3x3, self).__init__()
        self.sub_channel = in_channels // 3
        groups = self.sub_channel
        self.deform_conv3 = DeformConv(self.sub_channel, groups=groups, kernel_size=(3,3), padding=1)
        self.deform_conv5 = DeformConv(self.sub_channel, groups=groups, kernel_size=(5,5), padding=2)
        self.deform_conv7 = DeformConv(self.sub_channel, groups=groups, kernel_size=(7,7), padding=3)
        self.se_block = ChannelAttention(in_channels)


    def forward(self, x):
        # Split the input tensor along the channel dimension into 3 equal parts
        c3, c5, c7 = torch.chunk(x, 3, dim=1)

        # Apply deformable convolution with different kernel sizes
        out3 = self.deform_conv3(c3)
        out5 = self.deform_conv5(c5)
        out7 = self.deform_conv7(c7)

        # Concatenate the results along the channel dimension
        out = torch.cat([out3, out5, out7], dim=1)
        se_weight = self.se_block(out)
        out = out * se_weight  # 残差连接，保留原始信息
        return out


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        assert kernel_size in (3, 7), 'kernel must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MSGA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=2):
        super(MSGA, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 and self.in_channels == self.out_channels else False

        ex_channels = int(self.in_channels * expansion_factor)
        
        if ex_channels % 3 != 0:
            ex_channels = (ex_channels // 3 + 1) * 3

        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ex_channels),
            nn.GELU()
        )
        self.ms_deform_conv = MultiScaleDeformConv_3x3(ex_channels)
        self.pconv2 = nn.Sequential(
            nn.Conv2d(ex_channels, self.out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x):
        identity = x
        
        out = self.pconv1(x)
        out = self.ms_deform_conv(out) 
        out = self.pconv2(out)
        
        if self.use_skip_connection:
            return identity + out
        else:
            return out
        
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(EUCB,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.GELU()
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ) 
    def forward(self, x):
        x = self.up_dwc(x)
        # channel_shuffle might be beneficial, but let's keep it simple for now
        x = self.pwc(x)
        return x
        
    

class XcoddDecoder_Enhance_bottom(nn.Module):
    def __init__(self, channels=[384, 192, 96, 48]):
        super(XcoddDecoder_Enhance_bottom, self).__init__()
        # Copied from EMCAD_decoder.py and adapted

        
        eucb_ks = 3
        
        self.bottom_ca = ChannelAttention(channels[0], ratio=8)  # 减小压缩比，保持更多特征
        self.bottom_sa = SAB(kernel_size=7)
        # 添加门控机制，让模型学会是否使用底层增强
        self.bottom_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[0], 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.mscb4 = MSGA(channels[0], channels[0], stride=1)
        
        
        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.ccm3 = CGSC(channels[1],kernel_size=7)
        self.mscb3 = MSGA(channels[1], channels[1], stride=1)

        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.ccm2 = CGSC(channels[2],kernel_size=9)
        self.mscb2 = MSGA(channels[2], channels[2], stride=1)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.ccm1 = CGSC(channels[3],kernel_size=11)
        self.mscb1 = MSGA(channels[3], channels[3], stride=1)
        
    def forward(self, x, skips):
        
        gate_weight = self.bottom_gate(x)
        x = self.bottom_ca(x) * x
        x = self.bottom_sa(x) * x
        d4 = self.mscb4(x)
        d4 = x + gate_weight * (d4 - x)

        d3_up = self.eucb3(d4)
        x3_enhanced = self.ccm3(x_up=d3_up, x_skip=skips[0])
        d3 = x3_enhanced + d3_up
        d3 = self.mscb3(d3)

        d2_up = self.eucb2(d3)
        x2_enhanced = self.ccm2(x_up=d2_up, x_skip=skips[1])
        d2 = x2_enhanced + d2_up
        d2 = self.mscb2(d2)

        d1_up = self.eucb1(d2)
        x1_enhanced = self.ccm1(x_up=d1_up, x_skip=skips[2])
        d1 = x1_enhanced + d1_up
        d1 = self.mscb1(d1)
        
        return [d4, d3, d2, d1]