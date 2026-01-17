import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class FFM(nn.Module):
    """
    Frequency Focus Module (频率聚焦模块)
    作用：在频域中对特征进行滤波和加权，保留关键的频率分量。
    """
    def __init__(self, dim) -> None:
        super().__init__()
        # 使用 1x1 卷积进行特征变换，不改变空间尺寸
        # 这里为了保持频域处理的纯粹性，不使用 BN 和 Act，直接用 nn.Conv2d
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        
        # 可学习的频域加权参数 alpha 和 beta
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # 两个分支处理
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        # 快速傅里叶变换 (FFT) 转到频域 (dim=(-2, -1) 指定空间维度)
        x2_fft = torch.fft.fft2(x2, norm='backward')

        # 频域特征融合 (复数乘法)
        out = x1 * x2_fft

        # 逆傅里叶变换 (IFFT) 回到空域
        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        # 取模值，恢复实数特征
        out = torch.abs(out)

        # 加权残差连接
        return out * self.alpha + x * self.beta


class ImprovedFFTKernel(nn.Module):
    """
    结合多尺度空间卷积和频域注意力，增强对小目标的感知能力。
    """
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31 # 大卷积核尺寸，用于捕获长距离依赖
        pad = ker // 2
        
        # 输入投影层
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU() # 论文中使用 GELU
        )
        # 输出投影层
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        
        # 多尺度深度卷积 (Depthwise Conv)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.SiLU() # 激活函数

        # SCA (Spatial Context Attention) 部分的多尺度卷积
        # 用于在频域增强后提取不同感受野的空间特征
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)

        # 频域增强辅助分支
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ffm = FFM(dim) # 调用上面的 FFM 模块

        # 通道注意力机制 (Channel Attention)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. 初始特征变换
        out = self.in_conv(x)
        
        # 2. 频域增强 (对应论文公式 1)
        # 利用全局池化获取全局上下文
        x_att = self.fac_conv(self.fac_pool(out)) 
        # FFT 变换
        x_fft = torch.fft.fft2(out, norm='backward')
        # 频域加权
        x_fft = x_att * x_fft 
        # IFFT 逆变换
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)
        
        # 3. 多尺度空间特征提取 (对应论文公式 2)
        x_sca1 = self.conv1x1(x_fca)
        x_sca2 = self.conv3x3(x_fca)
        x_sca3 = self.conv5x5(x_fca)
        x_sca = x_sca1 + x_sca2 + x_sca3

        # 4. 通道注意力加权
        channel_weights = self.channel_attention(x_att)
        x_sca = x_sca * channel_weights

        # 5. FFM 模块进一步处理 (对应论文公式 3)
        x_sca = self.ffm(x_sca)

        # 6. 最终融合 (对应论文公式 4)
        # 结合原始特征、大核卷积特征和频域增强特征
        out = x + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        
        return self.out_conv(out)


class FFFE(nn.Module): 

    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e # 通道分割比例，默认 0.25 的通道进入频域处理分支
        
        # 使用 ConvNormLayer 以保持与主干网络一致的归一化方式
        # 注意：UAV-DETR 默认使用 SiLU，如果环境不支持可改为 'relu'
        self.cv1 = ConvNormLayer(dim, dim, filter_size=1, stride=1, act='silu')
        self.cv2 = ConvNormLayer(dim, dim, filter_size=1, stride=1, act='silu')
        
        # 核心频域内核只处理一部分通道，减少计算量，提升效率
        self.m = ImprovedFFTKernel(int(dim * self.e))

    def forward(self, x):
        # 1. 通道分割策略 (Split-Transform-Merge)
        c1 = round(x.size(1) * self.e)
        c2 = x.size(1) - c1
        
        # 2. 初始卷积
        cv1_out = self.cv1(x)
        
        # 3. 分割为两个分支
        # ok_branch: 进行频域增强
        # identity:  保持原样
        ok_branch, identity = torch.split(cv1_out, [c1, c2], dim=1)
        
        # 4. 合并分支并进行最终卷积
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))


# ==========================================
#  以下为原有代码 (保持原样)
# ==========================================

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size, stride, groups=1, act=None):
        super(ConvNormLayer, self).__init__()
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias=False)
        self.norm = nn.BatchNorm2d(ch_out)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', att=False):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_module('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act='relu')
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = out + short
        out = F.relu(out)
        return out

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d', att=False):
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

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = F.relu(out)
        return out

class GMKI(nn.Module):
    """
    Global Multi-Kernel Interaction (GMKI) Unit - 全局多核交互模块 
    """
    expansion = 1  
    def __init__(self, ch_in, ch_out=None, stride=1, shortcut=True, act='relu', variant='b', att=False):
        super(GMKI, self).__init__()
        # 兼容处理: 如果未指定 ch_out，默认输入输出通道保持一致
        out_channels = ch_out if ch_out is not None else ch_in
        
        # --- Stage 1: 局部感知 (Local Perception) ---
        # 使用 3x3 卷积捕捉细微的局部特征（如物体的边缘、纹理）
        self.local_perception = nn.Sequential(
            # Depthwise Conv: 提取空间特征，groups=ch_in 独立处理每个通道
            ConvNormLayer(ch_in, ch_in, filter_size=3, stride=stride, groups=ch_in, act='relu'),
            # Pointwise Conv: 1x1 卷积进行通道间的信息交互与融合
            ConvNormLayer(ch_in, ch_in, filter_size=1, stride=1, act='relu')
        )
        
        # --- Stage 2: 中域感知 (Medium Perception) ---
        # 使用 5x5 卷积扩大感受野，捕捉中等尺度的部件特征
        # 输入来自于 Stage 1 的输出，实现了特征的进一步抽象
        self.medium_perception = nn.Sequential(
            ConvNormLayer(ch_in, ch_in, filter_size=5, stride=1, groups=ch_in, act='relu'),
            ConvNormLayer(ch_in, ch_in, filter_size=1, stride=1, act='relu')
        )
        
        # --- Stage 3: 全局感知 (Global Perception) ---
        # 使用 7x7 大卷积核捕捉广泛的上下文背景信息（Context）
        # 这对于区分小目标和背景噪声至关重要
        self.global_perception = nn.Sequential(
            ConvNormLayer(ch_in, ch_in, filter_size=7, stride=1, groups=ch_in, act='relu'),
            # 最后一层 PW 卷积负责将通道数调整为最终需要的 out_channels
            ConvNormLayer(ch_in, out_channels, filter_size=1, stride=1, act='relu')
        )
        
        # --- 残差连接 (Shortcut) ---
        # 如果输入输出通道数不一致（例如用于改变维度的层），需要用 1x1 卷积对输入进行投影
        if stride != 1 or ch_in != out_channels:
            self.shortcut = ConvNormLayer(ch_in, out_channels, filter_size=1, stride=stride, act=None)
        elif shortcut:
            # 如果维度一致，直接恒等映射 (Identity)，不增加计算量
            self.shortcut = nn.Identity()
        else:
            self.shortcut = lambda x: 0 # 如果强制关闭 shortcut

    def forward(self, x):
        # 1. 保留原始输入，准备做残差连接
        identity = self.shortcut(x)
        
        # 2. 级联前向传播 (Cascade Forward)
        # 特征图依次通过小、中、大三种感受野的过滤
        x = self.local_perception(x)   # 捕捉细节
        x = self.medium_perception(x)  # 聚合形状
        x = self.global_perception(x)  # 感知背景
        
        # 3. 残差融合
        # 将提取到的多尺度深度特征与原始特征相加，防止梯度消失，增强特征复用
        return x + identity

class Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, *args, att=False, variant='b', use_fffe=False):
        super(Blocks, self).__init__()
        self.blocks = nn.ModuleList()
        
        # --- 参数解析 ---
        arg_list = list(args)
        if len(arg_list) > 0 and isinstance(arg_list[0], int):
             if len(arg_list) > 1 and not isinstance(arg_list[1], int):
                 arg_list.pop(0) 
             elif len(arg_list) > 1 and isinstance(arg_list[1], str):
                 arg_list.pop(0) 

        try:
            block_name = arg_list[0]
            count = arg_list[1]
            stage_num = arg_list[2]
            
            if len(arg_list) > 3: att = arg_list[3]
            if len(arg_list) > 4: variant = arg_list[4]
            if len(arg_list) > 5:
                if isinstance(arg_list[-1], bool): 
                    use_fffe = arg_list[-1]
                else:
                    use_fffe = arg_list[5]
            
        except IndexError:
            raise ValueError(f"Blocks 参数不匹配: {args}")

        self.use_fffe = use_fffe
        self.stage_num = stage_num  # 保存 stage_num 用于打印

        if isinstance(block_name, str):
            if block_name == 'BasicBlock':
                block_class = BasicBlock
            elif block_name == 'BottleNeck':
                block_class = BottleNeck
            elif block_name == 'GMKI':
                block_class = GMKI
            else:
                block_class = block_name
        else:
            block_class = block_name

        current_ch_in = ch_in
        for i in range(count):
            s = 2 if i == 0 and int(stage_num) != 2 else 1
            
            self.blocks.append(
                block_class(
                    ch_in=current_ch_in,
                    ch_out=ch_out,
                    stride=s,
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    att=att)
            )
            expansion = getattr(block_class, 'expansion', 1)
            current_ch_in = ch_out * expansion
        
        if self.use_fffe:
            self.fffe = FFFE(dim=current_ch_in)

    def forward(self, inputs):
        # --- [诊断打印] ---
        # 仅在初始批次打印，避免刷屏
        if not hasattr(self, 'printed_diag'):
            print(f"\n[DIAG] Stage {self.stage_num} Input: {inputs.shape}")
            
        block_out = inputs
        for i, block in enumerate(self.blocks):
            block_out = block(block_out)
            # 检查 BasicBlock 内部是否过度下采样
            if not hasattr(self, 'printed_diag') and i == 0:
                 print(f"[DIAG] Stage {self.stage_num} Block 0 Output: {block_out.shape}")
        
        if self.use_fffe:
            before_fffe = block_out.shape
            block_out = self.fffe(block_out)
            if not hasattr(self, 'printed_diag'):
                 print(f"[DIAG] Stage {self.stage_num} FFFE Change: {before_fffe} -> {block_out.shape}")
            
        if not hasattr(self, 'printed_diag'):
            print(f"[DIAG] Stage {self.stage_num} Final Output: {block_out.shape}")
            self.printed_diag = True # 标记已打印
            
        return block_out