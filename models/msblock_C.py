# Copyright (c) MCG-NKU. All rights reserved.
from typing import Sequence, Union
import torch
import torch.nn as nn
from torch import Tensor

from mmdet.utils import OptConfigType
from mmcv.cnn import ConvModule

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class MSBlockLayer(nn.Module):
    """MSBlockLayer

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int, tuple[int]): The kernel size of this Module.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to None.
    """
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: Union[int, Sequence[int]],
                 conv_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.in_conv = ConvModule(in_channel,
                                  out_channel,
                                  1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)        
        self.mid_conv = ConvModule(out_channel,
                                   out_channel,
                                   kernel_size,
                                   padding=autopad(kernel_size),
                                   groups=out_channel,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg,
                                   norm_cfg=norm_cfg)
        self.out_conv = ConvModule(out_channel,
                                   in_channel,
                                   1,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg, 
                                   norm_cfg=norm_cfg)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        x = self.in_conv(x)
        x = self.mid_conv(x)
        x = self.out_conv(x)
        return x
        


class MSBlock(nn.Module):
    """MSBlock

    Args:
        in_channel (int): The input channels of this Module.
        out_channel (int): The output channels of this Module.
        kernel_sizes (list(int, tuple[int])): Sequential of kernel sizes in MS-Block.
            
        in_expand_ratio (float): Channel expand ratio for inputs of MS-Block. Defaults to 3.
        mid_expand_ratio (float): Channel expand ratio for each branch in MS-Block. Defaults to 2.
        layers_num (int): Number of layer in MS-Block. Defaults to 3.
        in_down_ratio (float): Channel down ratio for downsample conv layer in MS-Block. Defaults to 1.
        
        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in MS-Block. Defaults to None.
        
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to dict(type='SiLU', inplace=True).
    """
    def __init__(self, 
                 in_channel: int,
                 out_channel: int,
                 kernel_sizes= [(3, 3), (5, 5)], #PAFPN 默认卷积核为3 kernel_sizes: Sequence[Union[int, Sequence[int]]],
                 
                 in_expand_ratio: float = 3.,
                 mid_expand_ratio: float = 2.,
                 layers_num: int = 3,
                 in_down_ratio: float = 1.,

                 conv_cfg: OptConfigType = None, 
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 ) -> None:
        super().__init__()
                
        self.in_channel = int(in_channel*in_expand_ratio//in_down_ratio)
        self.mid_channel = self.in_channel//len(kernel_sizes)
        self.mid_expand_ratio = mid_expand_ratio
        groups = int(self.mid_channel*self.mid_expand_ratio)
        self.layers_num = layers_num

        self.in_conv = ConvModule(in_channel,
                                  self.in_channel,
                                  1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)
        
        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(self.mid_channel,
                                    groups,
                                    kernel_size=kernel_size,
                                    conv_cfg=conv_cfg,
                                    act_cfg=act_cfg,
                                    norm_cfg=norm_cfg) for _ in range(int(self.layers_num))]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = ConvModule(self.in_channel,
                                   out_channel,
                                   1,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg,
                                   norm_cfg=norm_cfg)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        out = self.in_conv(x)
        channels = []
        for i, mid_conv in enumerate(self.mid_convs):
            channel = out[:, i*self.mid_channel:(i+1)*self.mid_channel,...]
            if i >= 1:
                channel = channel + channels[i-1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        return out


class MSBlock1357(nn.Module):
    """MSBlock
    paper：Rethinking Multi-Scale Representation Learning for Real-Time Object Detection
    https://github.com/open-mmlab/mmyolo
    Args:
        in_channel (int): The input channels of this Module.
        out_channel (int): The output channels of this Module.
        kernel_sizes (list(int, tuple[int])): Sequential of kernel sizes in MS-Block.

        in_expand_ratio (float): Channel expand ratio for inputs of MS-Block. Defaults to 3.
        mid_expand_ratio (float): Channel expand ratio for each branch in MS-Block. Defaults to 2.
        layers_num (int): Number of layer in MS-Block. Defaults to 3.
        in_down_ratio (float): Channel down ratio for downsample conv layer in MS-Block. Defaults to 1.

        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in MS-Block. Defaults to None.

        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_sizes=[1, (3, 3), (5, 5),(7,7)],  # PAFPN 默认卷积核为3 kernel_sizes: Sequence[Union[int, Sequence[int]]],

                 in_expand_ratio: float = 3.,
                 mid_expand_ratio: float = 2.,
                 layers_num: int = 3,
                 in_down_ratio: float = 1.,

                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 ) -> None:
        super().__init__()

        self.in_channel = int(in_channel * in_expand_ratio // in_down_ratio)
        self.mid_channel = self.in_channel // len(kernel_sizes)
        self.mid_expand_ratio = mid_expand_ratio
        groups = int(self.mid_channel * self.mid_expand_ratio)
        self.layers_num = layers_num

        self.in_conv = ConvModule(in_channel,
                                  self.in_channel,
                                  1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)

        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(self.mid_channel,
                                      groups,
                                      kernel_size=kernel_size,
                                      conv_cfg=conv_cfg,
                                      act_cfg=act_cfg,
                                      norm_cfg=norm_cfg) for _ in range(int(self.layers_num))]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = ConvModule(self.in_channel,
                                   out_channel,
                                   1,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg,
                                   norm_cfg=norm_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        out = self.in_conv(x)
        channels = []
        for i, mid_conv in enumerate(self.mid_convs):
            channel = out[:, i * self.mid_channel:(i + 1) * self.mid_channel, ...]
            if i >= 1:
                channel = channel + channels[i - 1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        return out


import torch.nn.functional as F

class MSBlock13579(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_sizes=[1, 3, 5, 7, 9],
                 in_expand_ratio: float = 3.,
                 mid_expand_ratio: float = 2.,
                 layers_num: int = 3,
                 in_down_ratio: float = 1.,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 ) -> None:
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)
        self.in_channel = int(in_channel * in_expand_ratio // in_down_ratio)

        # Fix rounding issue
        base_mid = self.in_channel // self.num_branches
        self.mid_channels = [base_mid] * self.num_branches
        self.mid_channels[0] += self.in_channel - sum(self.mid_channels)  # make total match
        assert sum(self.mid_channels) == self.in_channel

        self.mid_expand_ratio = mid_expand_ratio
        self.layers_num = layers_num

        self.in_conv = ConvModule(
            in_channel,
            self.in_channel,
            1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

        self.mid_convs = nn.ModuleList()
        for i, kernel_size in enumerate(kernel_sizes):
            mid_ch = self.mid_channels[i]
            groups = int(mid_ch * self.mid_expand_ratio)

            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
            else:
                layers = [
                    MSBlockLayer(
                        mid_ch,
                        groups,
                        kernel_size=kernel_size,
                        conv_cfg=conv_cfg,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg
                    ) for _ in range(layers_num)
                ]
                self.mid_convs.append(nn.Sequential(*layers))

        self.out_conv = ConvModule(
            self.in_channel,
            out_channel,
            1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.in_conv(x)
        channels = []
        start = 0
        # for i, mid_conv in enumerate(self.mid_convs):
        #     end = start + self.mid_channels[i]
        #     channel = out[:, start:end, ...]
        #     if i >= 1:
        #         channel = channel + channels[i - 1]
        #     channel = mid_conv(channel)
        #     channels.append(channel)
        #     start = end
        for i, mid_conv in enumerate(self.mid_convs):
            end = start + self.mid_channels[i]
            channel = out[:, start:end, ...]
            # ✅ 删除这行：跨通道相加
            # if i >= 1:
            #     channel = channel + channels[i - 1]
            channel = mid_conv(channel)
            channels.append(channel)
            start = end
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        return out


if __name__ == "__main__":
    # model = MSBlock(80, 160, [1, (3, 3), (3, 3)]) # 3个卷积核尺寸
    model = MSBlock(80, 160)  # 3个卷积核尺寸
    x= torch.rand(16,80,40,40)
    y = model(x)
    print()


