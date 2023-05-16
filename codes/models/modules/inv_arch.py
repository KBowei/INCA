import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .guassian_mixture import GaussianMixture
from .color_conversion import rgb2yuv, yuv2rgb
from .utils import channel2batch, batch2channel
import pdb


class VanillaInvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(VanillaInvBlock, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class EnhancedInvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(EnhancedInvBlock, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.E = subnet_constructor(self.split_len2, self.split_len1)
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            self.LR = x1 + self.E(x2)
            y1 = self.LR - self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            self.LR = x1 + self.F(y2)
            y1 = self.LR - self.E(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:  # froward pass
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:  # backward pass
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac
    
class learn_wavelet(torch.nn.Module):
    def __init__(self, subnet_constructor=None):
        super(learn_wavelet, self).__init__()
        self.lifting = subnet_constructor(channel_in=1, channel_out=1)
        self.color_conversion = rgb2yuv
        self.inverse_color_conversion = yuv2rgb
    def forward(self, x, rev=False):
        batch_size = x.size()[0]
        channel_in = x.size()[1]
        if not rev:
            if channel_in == 3:
             # [b,3,h,w] -> [3b, 1, h, w] rgb2yuv
                x = self.color_conversion(x)
            else:
                x = channel2batch(x, channel_in)
            # transform for rows
            L = x[:,:,0::2,:]
            H = x[:,:,1::2,:]
            L, H = self.lifting.forward(L, H)

            L = L.permute(0,1,3,2)
            LL = L[:,:,0::2,:]
            HL = L[:,:,1::2,:]
            LL, HL = self.lifting.forward(LL, HL)
            LL = LL.permute(0,1,3,2)
            HL = HL.permute(0,1,3,2)

            H = H.permute(0,1,3,2)
            LH = H[:,:,0::2,:]
            HH = H[:,:,1::2,:]
            LH, HH = self.lifting.forward(LH, HH)
            LH = LH.permute(0,1,3,2)
            HH = HH.permute(0,1,3,2)
            # [3b, 1, h/2, w/2] -> [b, 3, h/2, w/2]
            LL = batch2channel(LL, channel_in)
            HL = batch2channel(HL, channel_in)
            LH = batch2channel(LH, channel_in)
            HH = batch2channel(HH, channel_in)
            # return [b, 12, h/2, w/2]
            return torch.cat((LL, HL, LH, HH), dim=1)
        else:
            # [b, 12, h/2, w/2] -> [b, 3, h/2, w/2]
            LL, HL, LH, HH = torch.chunk(x, 4, dim=1)
            channel_in  = channel_in//4
            # [b, 3, h/2, w/2] -> [3b, 1, h/2, w/2]
            LL = channel2batch(LL, channel_in)
            HL = channel2batch(HL, channel_in)
            LH = channel2batch(LH, channel_in)
            HH = channel2batch(HH, channel_in)
            LH = LH.permute(0, 1, 3, 2)
            HH = HH.permute(0, 1, 3, 2)
            H = torch.zeros(LH.size()[0], LH.size()[1], LH.size()[2] + HH.size()[2], LH.size()[3], device=LH.device)
            LH, HH = self.lifting.forward(LH, HH, rev=True)
            H[:, :, 0::2, :] = LH
            H[:, :, 1::2, :] = HH
            H = H.permute(0, 1, 3, 2)

            LL = LL.permute(0, 1, 3, 2)
            HL = HL.permute(0, 1, 3, 2)
            L = torch.zeros(LL.size()[0], LL.size()[1], LL.size()[2] + HL.size()[2], LL.size()[3], device=LH.device)
            LL, HL = self.lifting.forward(LL, HL, rev=True)
            L[:, :, 0::2, :] = LL
            L[:, :, 1::2, :] = HL
            L = L.permute(0, 1, 3, 2)

            L, H = self.lifting.forward(L, H, rev=True)
            x = torch.zeros(L.size()[0], L.size()[1], L.size()[2] + H.size()[2], L.size()[3], device=LH.device)
            x[:, :, 0::2, :] = L
            x[:, :, 1::2, :] = H
            
            x = self.inverse_color_conversion(torch.cat((x[0:batch_size], x[batch_size:2*batch_size], x[2*batch_size:3*batch_size]), dim=1))

        return x

class SAINet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, e_blocks=5, v_blocks=3, down_num=2, gmm_components=5):
        super(SAINet, self).__init__()
        current_channel = channel_in

        operations = []
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
        self.haar_operations = nn.ModuleList(operations)

        operations = []
        for j in range(e_blocks):
            b = EnhancedInvBlock(subnet_constructor, current_channel, channel_out)
            operations.append(b)
        self.down_operations = nn.ModuleList(operations)
        
        operations = []
        for k in range(v_blocks):
            b = VanillaInvBlock(subnet_constructor, current_channel, channel_out)
            operations.append(b)
        self.comp_operations = nn.ModuleList(operations)
        
        # gaussian mixture model
        self.gmm = GaussianMixture(gmm_components)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0  # 见附录A, 因为flow-based model通常需要最大化输入分布的似然进行优化，本文训练时没有用到最大似然

        if not rev:  # rev=False when forward process
            for op in self.haar_operations:
                out = op.forward(out, rev)  # downsample, (b,c,h,w) -> (b,c*4,h//4,w//4)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            for op in self.down_operations:  # invertible blocks 
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            for op in self.comp_operations:  # compression simulator
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.comp_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            for op in reversed(self.down_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            for op in reversed(self.haar_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, self.down_operations[-1].LR, jacobian
        else:
            return out, self.down_operations[-1].LR

class HarrNet(nn.Module):
    """Harr小波实现
    input: (b,3,h,w)
    output: (b,12,h//4,w//4)  根据down_num调整
    
    """
    def __init__(self, channel_in=3, down_num=2, gmm_components=5):
        super(HarrNet, self).__init__()
        current_channel = channel_in

        operations = []
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
        self.haar_operations = nn.ModuleList(operations)
        
        # gaussian mixture model, 可以用来替代高频子带信息 (optional)
        self.gmm = GaussianMixture(gmm_components)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0  # 见SAIN附录A, 因为flow-based model通常需要最大化输入分布的似然进行优化，本文训练时没有用到最大似然，但是增加了该选项

        if not rev:  # forward pass
            for op in self.haar_operations:
                out = op.forward(out, rev)  # downsample, (b,c,h,w) -> (b,c*4,h//4,w//4)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else: # backward pass
            for op in reversed(self.haar_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out
    
class WaveNet(nn.Module):
    """实现类小波变换
    input: (b,3,h,w)
    output: (b,12,h//4,w//4)  根据down_num调整
    
    """
    def __init__(self, channel_in, subnet_constructor, down_num=1, gmm_components=5):
        super(WaveNet, self).__init__()
        current_channel = channel_in
        operations = []
        for i in range(down_num):
            b = learn_wavelet(subnet_constructor)
            operations.append(b)
            current_channel *= 4
        self.haar_operations = nn.ModuleList(operations)
        
        # gaussian mixture model, 可以用来替代高频子带信息 (optional)
        self.gmm = GaussianMixture(gmm_components)

    def forward(self, x, rev=False, cal_jacobian=False, color_conversion=False):
        jacobian = 0  # 见SAIN附录A, 因为flow-based model通常需要最大化输入分布的似然进行优化，本文训练时没有用到最大似然，但是增加了该选项
        out = x
        if not rev:  # forward pass
            for op in self.haar_operations:
                out = op.forward(out, rev)  # downsample, (b,c,h,w) -> (b,c*4,h//4,w//4)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else: # backward pass
            for op in reversed(self.haar_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out
