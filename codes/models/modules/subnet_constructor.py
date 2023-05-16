import models.modules.module_util as mutil
import torch
import torch.nn as nn
from torch.nn import functional as F


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5
    
class ResBlock(torch.nn.Module):
    def __init__(self, internal_channel=16):
        super(ResBlock, self).__init__()

        self.internal_channel = internal_channel
        self.padding = torch.nn.ReflectionPad2d(1)
        self.conv1 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(self.padding(x))
        out = self.relu(out)
        out = self.conv2(self.padding(out))
        return x + out


class P_block(torch.nn.Module):
    def __init__(self, channel_in=1, channel_out=1):
        super(P_block, self).__init__()
        self.padding_reflect = torch.nn.ReflectionPad2d(1)
        self.conv_pre = torch.nn.Conv2d(channel_in, 16, 3, 1, 0)  # 没有初始化
        self.res1 = ResBlock(16)
        self.res2 = ResBlock(16)
        self.conv_post = torch.nn.Conv2d(16, channel_out, 3, 1, 0)

    def forward(self, x):
        x = self.conv_pre(self.padding_reflect(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv_post(self.padding_reflect(x))
        return x

lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781, 1.149604398860241] # bior4.4
class learn_lifting97(torch.nn.Module):
    def __init__(self, channel_in=1, channel_out=1):
        super(learn_lifting97, self).__init__()
        self.leran_wavelet_rate = 0.1
        self.skip1 = torch.nn.Conv2d(channel_in, 1, (3, 1), padding=0, bias=False)
        self.skip1.weight = torch.nn.Parameter(torch.Tensor([[[[0.0], [lifting_coeff[0]], [lifting_coeff[0]]]]]),
                                               requires_grad=False)
        self.p_block1 = P_block(channel_in, 1)

        self.skip2 = torch.nn.Conv2d(1, 1, (3, 1), padding=0, bias=False)
        self.skip2.weight = torch.nn.Parameter(torch.Tensor([[[[lifting_coeff[1]], [lifting_coeff[1]], [0.0]]]]),
                                               requires_grad=False)
        self.p_block2 = P_block(1, 1)

        self.skip3 = torch.nn.Conv2d(1, 1, (3, 1), padding=0, bias=False)
        self.skip3.weight = torch.nn.Parameter(torch.Tensor([[[[0.0], [lifting_coeff[2]], [lifting_coeff[2]]]]]),
                                               requires_grad=False)
        self.p_block3 = P_block(1, 1)

        self.skip4 = torch.nn.Conv2d(1, 1, (3, 1), padding=0, bias=False)
        self.skip4.weight = torch.nn.Parameter(torch.Tensor([[[[lifting_coeff[3]], [lifting_coeff[3]], [0.0]]]]),
                                               requires_grad=False)
        self.p_block4 = P_block(1, channel_out)

        self.n_h = 0.0
        self.n_l = 0.0

    def forward(self, L, H, rev=False):
        if not rev:
            paddings = (0, 0, 1, 1)  

            tmp = F.pad(L, paddings, "reflect")
            skip1 = self.skip1(tmp)
            L_net = self.p_block1(skip1)
            H = H + skip1 + L_net * self.leran_wavelet_rate

            tmp = F.pad(H, paddings, "reflect")
            skip2 = self.skip2(tmp)
            H_net = self.p_block2(skip2)
            L = L + skip2 + H_net * self.leran_wavelet_rate

            tmp = F.pad(L, paddings, "reflect")
            skip3 = self.skip3(tmp)
            L_net = self.p_block3(skip3)
            H = H + skip3 + L_net * self.leran_wavelet_rate

            tmp = F.pad(H, paddings, "reflect")
            skip4 = self.skip4(tmp)
            H_net = self.p_block4(skip4)
            L = L + skip4 + H_net * self.leran_wavelet_rate

            H = H * (lifting_coeff[4] + self.n_h * self.leran_wavelet_rate)
            L = L * (lifting_coeff[5] + self.n_l * self.leran_wavelet_rate)

            return L, H
        else:
            H = H / (lifting_coeff[4] + self.n_h * self.leran_wavelet_rate)
            L = L / (lifting_coeff[5] + self.n_l * self.leran_wavelet_rate)

            paddings = (0, 0, 1, 1)

            tmp = F.pad(H, paddings, "reflect")
            skip4 = self.skip4(tmp)
            H_net = self.p_block4(skip4)
            L = L - skip4 - H_net * self.leran_wavelet_rate

            tmp = F.pad(L, paddings, "reflect")
            skip3 = self.skip3(tmp)
            L_net = self.p_block3(skip3)
            H = H - skip3 - L_net * self.leran_wavelet_rate

            tmp = F.pad(H, paddings, "reflect")
            skip2 = self.skip2(tmp)
            H_net = self.p_block2(skip2)
            L = L - skip2 - H_net * self.leran_wavelet_rate

            tmp = F.pad(L, paddings, "reflect")
            skip1 = self.skip1(tmp)
            L_net = self.p_block1(skip1)
            H = H - skip1 - L_net * self.leran_wavelet_rate

            return L, H


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'learn_lifting97':
            if channel_in == 1 and channel_out == 1:
                return learn_lifting97()
            else:
                print('The channel must be 1!')
                return None
    return constructor
