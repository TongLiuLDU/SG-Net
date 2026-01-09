"""
原生的像素焦点注意力机制
"""


# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def get_seqlen_and_mask(input_resolution, window_size):
    """
    计算序列长度和注意力掩码
    在边缘的时候，进行窗口注意力的时候，需要计算注意力掩码
    Args:
        input_resolution: 输入分辨率 [H, W]
        window_size: 窗口大小
    Returns:
        attn_local_length: 局部注意力序列长度
        attn_mask: 注意力掩码,用于标记填充位置
    """

    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]]), window_size,
                        dilation=1, padding=(window_size // 2, window_size // 2), stride=1)
    
    
    # 帮助模型了解每个局部窗口中有多少有效元素，这对于计算注意力权重是必要的。
    # 不难理解，其值只有4，6，9。在四个角落的时候，值为4，在边缘，值为6，在中心，值为9。
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)

    # 生成注意力掩码，用于标记填充位置.
    # 通过将attn_map的第一个维度（即batch维度）移除，并将其与第二个维度（即通道维度）进行转置，然后与0进行比较，生成一个布尔掩码。
    # 这个掩码用于标记注意力计算中哪些位置是填充的，从而在注意力计算中忽略这些填充位置。
    # 因为生成的矩阵便是torch.ones,所以等于0的便是填充位置
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask


class AggregatedAttention(nn.Module):
    """
    聚合注意力模块
    实现局部窗口注意力和全局池化注意力的融合,增强特征提取能力
    
    Args:
        dim: 输入特征维度
        input_resolution: 输入分辨率 [H, W]
        num_heads: 注意力头数
        window_size: 局部窗口大小
        qkv_bias: 是否使用偏置
        attn_drop: 注意力dropout率
        proj_drop: 投影层dropout率
        sr_ratio: 空间缩减比例，用于计算池化后的特征图尺寸
        fixed_pool_size: 固定池化大小
    """
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, fixed_pool_size=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # 基本参数设置
        self.dim = dim  # 输入特征维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = dim // num_heads  # 每个头的维度

        self.sr_ratio = sr_ratio  # 空间缩减比例

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size  # 局部窗口大小
        self.local_len = window_size ** 2  # 局部窗口内像素数

        # 设置池化参数
        if fixed_pool_size is None:
            self.pool_H, self.pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
        else:
            assert fixed_pool_size < min(input_resolution), \
                f"The fixed_pool_size {fixed_pool_size} should be less than the shorter side of input resolution {input_resolution} to ensure pooling works correctly."
            self.pool_H, self.pool_W = fixed_pool_size, fixed_pool_size
            
        self.pool_len = self.pool_H * self.pool_W  # 池化后特征长度

        # 特征展开层
        # 一个用于将输入特征图展开为滑动窗口块的操作，将输入特征图分割成多个小块（窗口），这些小块可以被视为局部区域的特征。
        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        
        # 温度参数,用于调节注意力分布的软度，通过引入可学习的温度参数，模型可以在训练过程中自动调整注意力分布的平滑程度，以适应不同的输入特征和任务需求。这种动态调整有助于提高模型的灵活性和性能。
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # 初始化softplus(temperature)为1/0.24

        # 查询层
        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # 查询变换
        self.query_embedding = nn.Parameter(  # 查询嵌入
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        
        
        
        # K,V层
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 键值变换
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力dropout
        self.proj = nn.Linear(dim, dim)  # 输出投影
        self.proj_drop = nn.Dropout(proj_drop)  # 投影dropout
        

        # 池化特征生成组件
        self.pool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))  # 自适应平均池化
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)  # 1x1卷积
        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.act = nn.GELU()  # GELU激活函数

        # 连续相对位置偏置生成MLP
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)  # 第一个全连接层
        self.cpb_act = nn.ReLU(inplace=True)  # ReLU激活
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)  # 第二个全连接层

        # 局部特征的相对位置偏置，nn.init.trunc_normal_用于初始化参数，mean=0, std=0.0004
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004))

        # 生成填充掩码和序列长度缩放
        local_seq_length, padding_mask = get_seqlen_and_mask(input_resolution, window_size)
        
        
        # 注册序列长度缩放和填充掩码
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(local_seq_length.numpy() + self.pool_len)),
                             persistent=False)          # 常数
        
        self.register_buffer("padding_mask", padding_mask, persistent=False)

        # 动态局部偏置
        self.learnable_tokens = nn.Parameter(  # 可学习的令牌
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))  # 可学习的偏置

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        """
        前向传播函数
        Args:
            x: 输入特征 [B, N, C]       B: 批量大小, N(HW): 序列长度, C: 特征维度
            H, W: 特征图高度和宽度
            relative_pos_index: 相对位置索引
            relative_coords_table: 相对坐标表
        Returns:
            x: 输出特征 [B, N, C]
        """
        B, N, C = x.shape

        # 生成查询向量,L2归一化,添加查询嵌入,然后通过序列长度缩放和温度参数调节
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)    # 将输入特征x通过线性变换和reshape操作，将特征维度从[B, N, C]转换为[B, N, num_heads, head_dim]，然后通过permute操作将维度顺序从[B, N, num_heads, head_dim]转换为[B, num_heads, N, head_dim]。最后，通过F.normalize函数对每个头内的特征进行L2归一化，确保每个头的特征向量具有相同的长度。dim=-1表示在最后一个维度上进行归一化。
        # 温度用于动态调整注意力分布的平滑程度，序列长度缩放是为了在计算注意力时考虑序列长度的影响，确保注意力机制在不同的序列长度上都能有效工作。
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * self.seq_length_scale    # 将归一化后的查询向量与查询嵌入相加，然后通过softplus函数对温度参数进行平滑处理，最后将结果与序列长度缩放相乘，得到最终的查询向量。

        # 生成展开的键值对并进行L2归一化
        k_local, v_local = self.kv(x).chunk(2, dim=-1)      # chunk(2, dim=-1)将键值对分成两部分，dim=-1表示在最后一个维度上进行分割。
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)     # 将键值对展开为局部窗口块，并进行维度重塑和转置，最后将键值对分成两部分，分别用于计算局部相似度和值。

        # 计算局部相似度
        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) \
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(self.padding_mask, float('-inf'))


        
        # 生成池化特征
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, self.pool_len).permute(0, 2, 1)                # 在这里使用1x1卷积将来代替Linear层
        x_ = self.norm(x_)

        # 生成池化键值对
        kv_pool = self.kv(x_).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        # 使用MLP生成池化特征的连续相对位置偏置
        pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                    relative_pos_index.view(-1)].view(-1, N, self.pool_len)
        
        # 计算池化相似度
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias


        # 拼接局部和池化相似度矩阵,通过同一个Softmax计算注意力权重
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 分离注意力权重,分别聚合局部和池化特征的值。即最后一部，乘以value
        attn_local, attn_pool = torch.split(attn, [self.local_len, self.pool_len], dim=-1)
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(
            -2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        # 线性投影和输出
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
