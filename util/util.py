import numpy as np
import torch

def prediction_concat(output1, output2, output3, anchors, classes, height, width):
    """
    #PS:默认放缩比 8, 16, 32
    :param output1: y1 层检测框
    :param output2: y2 层检测框
    :param output3: y3 层检测框
    :param anchors: default 框 (获取anchors的数量)
    :param classes: 类别数量
    :return:
    """
    # 获取三个y层anchor数量
    anchorNum1 = len(anchors[0])
    anchorNum2 = len(anchors[1])
    anchorNum3 = len(anchors[2])
    # 首先变换形式
    output1 = outputDeform(output1, anchorNum1, classes, height, width, 32)
    output2 = outputDeform(output2, anchorNum2, classes, height, width, 16)
    output3 = outputDeform(output3, anchorNum3, classes, height, width, 8)

    # 然后将三者进行合并 在dim = 1进行合并
    prediction = torch.cat([output1, output2, output3], 1)

    # 然后进行变换处理
    prediction[..., 0] = torch.sigmoid(prediction[..., 0])
    prediction[..., 1] = torch.sigmoid(prediction[..., 1])
    prediction[..., 4] = torch.sigmoid(prediction[..., 4])
    prediction[..., 5:-1] = torch.sigmoid(prediction[..., 5:-1])

    return prediction


def outputDeform(output, anchorsNumber, classes, height, width, stride):
    # stride: 特征图相对于原图的放缩比
    batch_size = output.size(0)
    height = height // stride
    width = width // stride
    output = output.view(batch_size, (classes + 5) * anchorsNumber, height*width)
    # transpose之后会不连续 所以要 contigous
    output = output.transpose(1,2).contiguous()
    output = output.view(batch_size, height*width*anchorsNumber, classes+5)
    return output
