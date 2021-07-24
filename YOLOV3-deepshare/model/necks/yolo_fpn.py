'''
neck部分 448输入时是56,28,14 416输入时为52,26,13
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.conv_module import Convolutional


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out


class FPN_YOLOV3(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """
    def __init__(self, fileters_in, fileters_out):
        super(FPN_YOLOV3, self).__init__()

        fi_0, fi_1, fi_2 = fileters_in
        fo_0, fo_1, fo_2 = fileters_out

        # large 输入：13*13*1024
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1,pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv0_0 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv0_1 = Convolutional(filters_in=1024, filters_out=fo_0, kernel_size=1,
                                       stride=1, pad=0)
        #输出 13*13*3*(20+5)

        #上采样准备与24*24*512的中等scale进行融合
        self.__conv0 = Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                                      activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route0 = Route()

        # medium 输入26*26*512
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fi_1+256, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv1_0 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv1_1 = Convolutional(filters_in=512, filters_out=fo_1, kernel_size=1,
                                       stride=1, pad=0)
        #输出 26*26*3*(20+5)

        #上采样，准备与52*52*256的小scale进行融合
        self.__conv1 = Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                                     activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route1 = Route()

        # small
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fi_2+128, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv2_0 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv2_1 = Convolutional(filters_in=256, filters_out=fo_2, kernel_size=1,
                                       stride=1, pad=0)


    def forward(self, x0, x1, x2):  # small 13x13x1024, medium 26x26x512, large52x52x256
        # large
        r0 = self.__conv_set_0(x0)
        out0 = self.__conv0_0(r0)
        out0 = self.__conv0_1(out0)

        # medium
        r1 = self.__conv0(r0)
        r1 = self.__upsample0(r1)
        x1 = self.__route0(x1, r1) # concat结果 26x26x(512+N)
        r1 = self.__conv_set_1(x1)
        out1 = self.__conv1_0(r1)
        out1 = self.__conv1_1(out1)

        # small
        r2 = self.__conv1(r1)
        r2 = self.__upsample1(r2)
        x2 = self.__route1(x2, r2) # 上采样再接拼接
        r2 = self.__conv_set_2(x2)
        out2 = self.__conv2_0(r2)
        out2 = self.__conv2_1(out2)

        return out2, out1, out0  # small, medium, large