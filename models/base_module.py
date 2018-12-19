import torch
import torch.nn as nn
import torch.nn.functional as F
from bases.block_base import BlockBase
import math


class ConvBlock(BlockBase):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=False, activation=None, norm=None, order='cba'):
        super(ConvBlock, self).__init__(in_channels, out_channels, activation, norm)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        operators = [self.conv, self.bn, self.act]
        order_dict = {'c': 0, 'b': 1, 'a': 2}
        split_words = list(order)
        avail_operators = []
        for c in split_words:
            op_id = order_dict[c]
            if operators[op_id] is not None:
                avail_operators.append(operators[op_id])
        self.avail_operators = nn.Sequential(*avail_operators)

    def forward(self, x):
        return self.avail_operators(x)

    def init_weights(self, weights):
        self.conv.weight.data.copy_(weights)


class DeconvBlock(BlockBase):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False,
                 activation=None, norm=None, order='cba'):
        super(DeconvBlock, self).__init__(in_channels, out_channels, activation, norm)
        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                               output_padding=output_padding, bias=bias)
        operators = [self.deconv, self.bn, self.act]
        order_dict = {'c': 0, 'b': 1, 'a': 2}
        split_words = list(order)
        avail_operators = []
        for c in split_words:
            op_id = order_dict[c]
            if operators[op_id] is not None:
                avail_operators.append(operators[op_id])
        self.avail_operators = nn.Sequential(*avail_operators)

    def forward(self, x):
        return self.avail_operators(x)

    def init_weights(self, weights):
        self.deconv.weight.data.copy_(weights)


class ResBlock(BlockBase):
    '''
    from Lim, Bee, et al. "Enhanced deep residual networks for single image super-resolution."
     The IEEE conference on computer vision and pattern recognition (CVPR) workshops. Vol. 1. No. 2. 2017.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, activation=None,
                 norm=None, order='cba', res_scale=1):
        super(ResBlock, self).__init__(in_channels, out_channels, activation, norm)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, activation=activation,
        #                        norm=norm, bias=bias, order=order)
        # self.conv2 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, activation=activation,
        #                        norm=norm, bias=bias, order=order)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.res_scale = res_scale

    def forward(self, x):
        residual = x
        out = self.conv1(x)

        if self.act is not None:
            out = self.act(out)

        out = self.conv2(out).mul(self.res_scale)
        out = torch.add(out, residual)
        return out


class InceptionBlock(BlockBase):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, activation='relu', norm='batch'):
        super(InceptionBlock, self).__init__(in_channels, out_channels, activation, norm)

        sub_channels1x1 = out_channels//3
        self.branch1x1 = ConvBlock(in_channels, sub_channels1x1, stride=stride, padding=0, kernel_size=1, bias=bias)

        sub_channels3x3 = out_channels // 3
        self.branch3x3da_1 = ConvBlock(in_channels, sub_channels3x3, padding=0, kernel_size=1, bias=bias)
        self.branch3x3da_2 = ConvBlock(sub_channels3x3, sub_channels3x3, kernel_size=kernel_size, padding=padding, bias=bias)

        sub_channels5x5 = out_channels - sub_channels1x1 - sub_channels3x3
        self.branch3x3dbl_1 = ConvBlock(in_channels, sub_channels5x5, padding=0, kernel_size=1)
        self.branch3x3dbl_2 = ConvBlock(sub_channels5x5, sub_channels5x5, kernel_size=kernel_size, padding=padding, bias=bias)
        self.branch3x3dbl_3 = ConvBlock(sub_channels5x5, sub_channels5x5, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch3x3da_1(x)
        branch5x5 = self.branch3x3da_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        # branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl]
        return torch.cat(outputs, 1)


# class PSBlock(BlockBase):
#     def __init__(self, in_channels, out_channels, upscale, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm='batch'):
#         super(PSBlock, self).__init__(in_channels, out_channels, activation, norm)
#         self.conv = torch.nn.Conv2d(in_channels, out_channels * upscale ** 2, kernel_size, stride, padding,
#                                     bias=bias)
#         self.ps = torch.nn.PixelShuffle(upscale)
#
#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.ps(self.conv(x)))
#         else:
#             out = self.ps(self.conv(x))
#
#         if self.activation is not None:
#             out = self.act(out)
#         return out

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(ConvBlock(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(ConvBlock(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False
