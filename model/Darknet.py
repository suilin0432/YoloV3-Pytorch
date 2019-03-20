import torch
import torch.nn as nn
import numpy as np
from util.util import *
# OrderedDict 是记住插入顺序的字典
from collections import OrderedDict
import math

__all__ = ["darknet21", "darknet53"]


# DarkNet中的基本
class ResUnit(nn.Module):
    def __init__(self, inplanes, planes):
        """

        :param inplanes: 输入channel
        :param planes: 每一层的输出channel
        """
        super(ResUnit, self).__init__()
        # 第一个DBL模块
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        # 第二个DBL模块
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        # 这个是为了做残差加法
        residual = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        return output + residual

#DarkNet
class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32 #输入到第一个ResUnit 时候的channel大小
        # 进入的第一个DBL模块
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        # 开始构造ResUnit模块
        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 获取所有的 模型中的层
        for m in self.modules():
            # 如果是卷积
            if isinstance(m, nn.Conv2d):
                # 卷积核所有参数进行一个随机初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm参数初始化
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        """

        :param planes: 每一层的channel数目
        :param blocks: 包含的ResUnit数目
        :return:
        """
        # 每个res大模块包含了 一个 padding 一个 DBL以及 layer[i] 个ResUnit
        layers = []
        # 下面三个是前面的DBL模块
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 更新当前的通道数目
        self.inplanes = planes[1]
        # 开始进行ResUnit的构建
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), ResUnit(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # 后面三个取出来之后给Yolo环节进行使用
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out3, out4, out5

def darknet21(pretrained, **kwargs):
    # 构建darknet21模型
    model = DarkNet([1,1,2,2,1])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet21 need the right pretrained path. got [{}]".format(pretrained))
    return model

def darknet53(pretrained, **kwargs):
    # 构建darknet21模型
    model = DarkNet([1,2,8,8,4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet53 need the right pretrained path. got [{}]".format(pretrained))
    return model