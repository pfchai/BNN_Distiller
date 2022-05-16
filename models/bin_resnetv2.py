'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function


__all__ = ['resnet18A_1w1a', 'resnet18B_1w1a', 'resnet18C_1w1a', 'resnet18_1w1a']

# binarized modules

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))

    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - w.mean([1, 2, 3], keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var([1, 2, 3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
        EW = torch.mean(torch.abs(w1))
        Q_tau = (- EW * torch.log(2 - 2 * self.tau)).detach().cpu().item()
        w2 = torch.clamp(w1, -Q_tau, Q_tau)

        if self.training:
            a0 = a / torch.sqrt(a.var([1, 2, 3], keepdim=True) + 1e-5)
        else:
            a0 = a

        # binarize
        bw = BinaryQuantize().apply(w2)
        ba = BinaryQuantize_a().apply(a0)
        # 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        # scaling factor
        output = output * self.alpha
        return output


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock_1w1a, self).__init__()
        self.is_last = is_last
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        pad = 0 if planes == self.expansion * in_planes else planes // 4
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d((2, 2)),
                LambdaLayer(lambda x: F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0))
            )

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.hardtanh(out, inplace=True)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck_1w1a(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_1w1a, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = F.hardtanh(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_channel[0]

        self.conv1 = nn.Conv2d(3, num_channel[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel[0])
        self.layer1 = self._make_layer(block, num_channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channel[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channel[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(num_channel[3] * block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(num_channel[3] * block.expansion)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            # layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _f(self, x, step):
        if step == 1:
            out = self.bn1(self.conv1(x))
        elif step == 2:
            out = self.layer1(x)
        elif step == 3:
            out = self.layer2(x)
        elif step == 4:
            out = self.layer3(x)
        elif step == 5:
            out = self.layer4(x)
        else:
            out = F.avg_pool2d(x, 4)
            out = out.view(out.size(0), -1)
            out = self.bn2(out)
            out = self.linear(out)
        return out

    def forward(self, x, steps=6):
        'step <= 6'
        for step in range(1, steps + 1):
            x = self._f(x, step)
        return x


def resnet18A_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2, 2, 2, 2], [32, 32, 64, 128], **kwargs)

def resnet18B_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2, 2, 2, 2], [32, 64, 128, 256], **kwargs)

def resnet18C_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2, 2, 2, 2], [64, 64, 128, 256], **kwargs)

def resnet18_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2, 2, 2, 2], [64, 128, 256, 512], **kwargs)

def resnet34_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)

def resnet50_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)

def resnet101_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3, 4, 23, 3], [64, 128, 256, 512], **kwargs)

def resnet152_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3, 8, 36, 3], [64, 128, 256, 512], **kwargs)


if __name__ == "__main__":
    net = resnet18_1w1a(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    feats, logit = net(x)
    for step in range(1, 7):
        out = net(x, step)
        print(out.shape)
