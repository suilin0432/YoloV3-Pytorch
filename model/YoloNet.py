import torch
import torch.nn as nn
import numpy as np
from util.modelChoose import backboneChoose
from util.util import *

class YoloNet(nn.Module):
    def __init__(self, config, isTraining = True):
        super(YoloNet, self).__init__()
        self.config = config
        self.isTraining = isTraining
        # 构建backbone
        self.module_params = config["model_params"]
        backbone_ = backboneChoose[self.module_params["backbone_name"]]
        self.backbone = backbone_(self.module_params["backbone_pretrained"])
        darknetFilters = self.backbone.layers_out_filters
        # yolochannel数目 (可能因为每个检测器设置的默认的anchor数量是不同的，所以会单独计算)
        yoloMassage = self.config["yolo"]
        anchors = yoloMassage["anchors"]
        classes = yoloMassage["classes"]
        yoloChannel1 = (classes + 5) * len(anchors[0])
        yoloChannel2 = (classes + 5) * len(anchors[1])
        yoloChannel3 = (classes + 5) * len(anchors[2])
        # 开始构建剩余的Yolo网络
        # yolo output1
        self.convolutionalSet1 = self.convolutionSetBuild([512, 1024], darknetFilters[-1])
        self.DBL1 = self.makeDBL(512, 1024, 3)
        # 输出就是 y1
        self.yoloConv1 = nn.Conv2d(1024, yoloChannel1, kernel_size=1, stride=1, padding=0, bias=True)

        # yolo output2
        self.DBL_DOWN1 = self.makeDBL(512, 256, 1)
        self.Upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        # 加上 256 是因为 concat了
        self.convolutionalSet2 = self.convolutionSetBuild([256, 512], darknetFilters[-2] + 256)
        self.DBL2 = self.makeDBL(256, 512, 3)
        # 输出就是 y2
        self.yoloConv2 = nn.Conv2d(512, yoloChannel2, kernel_size=1, stride=1, padding=0, bias=True)
        # yolo output3
        self.DBL_DOWN2 = self.makeDBL(256, 128, 1)
        self.Upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        # 加上 128 是因为 concat了
        self.convolutionalSet3 = self.convolutionSetBuild([128, 256], darknetFilters[-3] + 128)
        self.DBL3 = self.makeDBL(128, 256, 3)
        # 输出就是y3
        self.yoloConv3 = nn.Conv2d(256, yoloChannel3, kernel_size=1, stride=1, padding=0, bias=True)

    def makeDBL(self, inDim, outDim, kernelSize):
        pad = (kernelSize - 1) // 2 if kernelSize else 0
        return nn.Sequential(
            nn.Conv2d(inDim, outDim, kernel_size=kernelSize, padding=pad, bias=False),
            nn.BatchNorm2d(outDim),
            nn.LeakyReLU(0.1),
        )

    def convolutionSetBuild(self, Dims, inDim):
        return nn.Sequential(
            self.makeDBL(inDim, Dims[0], 1),
            self.makeDBL(Dims[0], Dims[1], 3),
            self.makeDBL(Dims[1], Dims[0], 1),
            self.makeDBL(Dims[0], Dims[1], 3),
            self.makeDBL(Dims[1], Dims[0], 1),
        )

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        convolutionOutput1 = self.convolutionalSet1(x0)
        output1 = self.DBL1(convolutionOutput1)
        output1 = self.yoloConv1(output1)
        convolutionOutput2 = self.DBL_DOWN1(convolutionOutput1)
        convolutionOutput2 = self.Upsample1(convolutionOutput2)
        convolutionOutput2 = torch.cat([convolutionOutput2, x1], 1)
        convolutionOutput2 = self.convolutionalSet2(convolutionOutput2)
        output2 = self.DBL2(convolutionOutput2)
        output2 = self.yoloConv2(output2)
        convolutionOutput3 = self.DBL_DOWN2(convolutionOutput2)
        convolutionOutput3 = self.Upsample2(convolutionOutput3)
        convolutionOutput3 = torch.cat([convolutionOutput3, x2], 1)
        convolutionOutput3 = self.convolutionalSet3(convolutionOutput3)
        output3 = self.DBL3(convolutionOutput3)
        output3 = self.yoloConv3(output3)
        # 将output1 ouput2 output3 的结构进行合并
        # 现在是 batchsize, anchorNums*(5+classesNum), height, width
        # 要合并为 batchsize, anchorNums * height * width, (5+classesNum)
        # 并且在其中会进行 各个点的sigmoid、等变换操作
        prediction = prediction_concat(output1, output2, output3, self.config["yolo"]["anchors"], self.config["yolo"]["classes"], self.config["img_h"], self.config["img_w"])
        return prediction

        # return output1, output2, output3

    def load_darknet_weights(self, weights_path):
        import numpy as np
        #Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values
        # Needed to write header when saving weights
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights
        print ("total len weights = ", weights.shape)
        fp.close()

        ptr = 0
        all_dict = self.state_dict()
        all_keys = self.state_dict().keys()
        print (all_keys)
        last_bn_weight = None
        last_conv = None
        for i, (k, v) in enumerate(all_dict.items()):
            if 'bn' in k:
                if 'weight' in k:
                    last_bn_weight = v
                elif 'bias' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_bias: ", ptr, num_b, k)
                    ptr += num_b
                    # weight
                    v = last_bn_weight
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_weight: ", ptr, num_b, k)
                    ptr += num_b
                    last_bn_weight = None
                elif 'running_mean' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_mean: ", ptr, num_b, k)
                    ptr += num_b
                elif 'running_var' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_var: ", ptr, num_b, k)
                    ptr += num_b
                    # conv
                    v = last_conv
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("conv wight: ", ptr, num_b, k)
                    ptr += num_b
                    last_conv = None
                else:
                    raise Exception("Error for bn")
            elif 'conv' in k:
                if 'weight' in k:
                    last_conv = v
                else:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("conv bias: ", ptr, num_b, k)
                    ptr += num_b
                    # conv
                    v = last_conv
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("conv wight: ", ptr, num_b, k)
                    ptr += num_b
                    last_conv = None
        print("Total ptr = ", ptr)
        print("real size = ", weights.shape)