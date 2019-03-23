import numpy as np
import torch
import cv2

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

# 图片输入时候的变换
def detection_img_input_convert(img, targetSize):
    # 图片变换缩放
    target_h, target_w = targetSize
    height, width = img.shape[:2]
    scale_w = target_w / width
    scale_h = target_h / height
    minScale = min(scale_h, scale_w)
    after_w = int(width * minScale)
    after_h = int(height * minScale)
    pad_w = target_w - after_w
    pad_h = target_h - after_h
    pad = ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0,0))
    img = cv2.resize(img, (after_w, after_h),interpolation = cv2.INTER_CUBIC)
    img = np.pad(img, pad, "constant", constant_values=128)
    # 转变图片为torch格式
    img = img[:, :, ::-1].copy()
    img = img/255.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

# 用来进行detection的时候获取检测框
def detection_anchors(prediction, CUDA, gridSize, anchors):
    # 要进行三个维度的grid的生成
    grid1 = gridshedGenerate(CUDA, gridSize[0], len(anchors[0]))
    grid2 = gridshedGenerate(CUDA, gridSize[1], len(anchors[1]))
    grid3 = gridshedGenerate(CUDA, gridSize[2], len(anchors[2]))
    # batchSize, numAnchor*width*height, numClass+5 在 dim=1 维度上进行合并
    grid = torch.cat([grid1, grid2, grid3], 1)
    prediction[...,0:2] += grid
    # 重复anchors
    anchors1 = anchorsRepeat(anchors[0], gridSize[0], 32)
    anchors2 = anchorsRepeat(anchors[1], gridSize[1], 16)
    anchors3 = anchorsRepeat(anchors[2], gridSize[2], 8)
    anchors_ = torch.cat([anchors1, anchors2, anchors3], 1)

    prediction[...,2:4] = torch.exp(prediction[...,2:4])
    prediction[...,2:4] *= anchors_

    newPrediction = torch.zeros(prediction[...,0:7].shape)
    newPrediction[...,0:5] = prediction[...,0:5]

    predictionClassesScore = prediction[...,5:]
    maxScore, maxIndex = torch.max(predictionClassesScore, -1)
    maxIndex = maxIndex.unsqueeze(0)
    # 最后两个维度记录confidenceScore 和 对应类别
    newPrediction[...,5] = maxScore
    newPrediction[...,6] = maxIndex


    mulBase1 = torch.Tensor([32])
    mulBase2 = torch.Tensor([16])
    mulBase3 = torch.Tensor([8])
    mul1 = mulBase1.repeat(3*gridSize[0]*gridSize[0], 1).unsqueeze(0)
    mul2 = mulBase2.repeat(3*gridSize[1]*gridSize[1], 1).unsqueeze(0)
    mul3 = mulBase3.repeat(3*gridSize[2]*gridSize[2], 1).unsqueeze(0)
    mul = torch.cat([mul1, mul2, mul3], 1)

    newPrediction[...,0:4] *= mul

    # 进行NMS处理
    newPrediction = NMS(newPrediction, conf_threshold=0.8, nms_threshold=0.5)

    return newPrediction



def gridshedGenerate(CUDA, gridSize, numAnchor):
    grid = np.arange(gridSize)
    x_offset, y_offset = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(x_offset).view(-1, 1)
    y_offset = torch.FloatTensor(y_offset).view(-1, 1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, numAnchor).view(-1, 2).unsqueeze(0)
    return x_y_offset

def anchorsRepeat(anchor, gridSize, scale):
    anchor = anchor / scale
    anchor = anchor.repeat(gridSize*gridSize, 1).unsqueeze(0)
    return anchor

def NMS(prediction, conf_threshold = 0.5, nms_threshold = 0.5):
    # PS: 长或宽小于0 的应该过滤掉吧

    # 首先过滤所有没有超过conf_threshold的项
    # 要对每个batch进行单独的操作以维持每个batch的形状

    # 改变 中心xy, 宽高 为 左上xy 右下xy 的格式
    b1_x1, b1_y1 = prediction[..., 0] - prediction[..., 2] / 2, prediction[..., 1] - prediction[..., 3] / 2
    b1_x2, b1_y2 = prediction[..., 0] + prediction[..., 2] / 2, prediction[..., 1] + prediction[..., 3] / 2
    prediction[..., 0] = b1_x1
    prediction[..., 1] = b1_y1
    prediction[..., 2] = b1_x2
    prediction[..., 3] = b1_y2
    totalPrediction = []
    for index in range(prediction.size(0)):
        batchPrediction = []
        pre = prediction[index]
        preHas = (pre[...,4] >= conf_threshold).squeeze()
        pre = pre[preHas]

        # 获取所有出现过的类别编号
        unique_labels = pre[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        # 对每个类别的框进行NMS操作
        for label in unique_labels:
            preClass = pre[pre[..., -1] == label]
            _, confSortedPre = torch.sort(preClass[:, 4], descending=True)
            preClass = preClass[confSortedPre]
            max_detections = []
            while preClass.size(0):
                # 添加当前类别当前分数最高的框
                max_detections.append(preClass[0].unsqueeze(0))
                # print(max_detections)
                # 如果只剩下当前这一个的时候直接终端就可以了
                if len(preClass) == 1:
                    break
                # 对上一个进入max_detections的bbox进行IoU的计算
                iou = bbox_iou(max_detections[-1], preClass[1:])
                # 将所有IoU重叠过多的框以及当前的框去掉
                preClass = preClass[1:][iou < nms_threshold]
            # 将list中的框整合成 torch Tensor变量
            # max_detections = torch.cat(max_detections).data
            # print(max_detections.shape)
            batchPrediction.extend(max_detections)
        if len(batchPrediction):
            batchPrediction = torch.cat(batchPrediction, 1)
        totalPrediction.append(batchPrediction)
    # exit()
    if len(totalPrediction[0]):
        totalPrediction = torch.cat(totalPrediction, 1)
    return totalPrediction


def bbox_iou(bbox1, bbox2):
    # 首先将bbox进行转换 从 中心, 宽高 -> 左上角, 右下角 的格式
    b1_x1, b1_y1 = bbox1[:, 0] - bbox1[:, 2] / 2, bbox1[:, 1] - bbox1[:, 3] / 2
    b1_x2, b1_y2 = bbox1[:, 0] + bbox1[:, 2] / 2, bbox1[:, 1] + bbox1[:, 3] / 2
    b2_x1, b2_y1 = bbox2[:, 0] - bbox2[:, 2] / 2, bbox2[:, 1] - bbox2[:, 3] / 2
    b2_x2, b2_y2 = bbox2[:, 0] + bbox2[:, 2] / 2, bbox2[:, 1] + bbox2[:, 3] / 2
    # 进行重叠区域的计算
    overlap_x1 = torch.max(b1_x1, b2_x1)
    overlap_x2 = torch.min(b1_x2, b2_x2)
    overlap_y1 = torch.max(b1_y1, b2_y1)
    overlap_y2 = torch.min(b1_y2, b2_y2)

    area = torch.clamp((overlap_x2 - overlap_x1), min=0) * torch.clamp((overlap_y2 - overlap_y1), min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = area / (b1_area + b2_area - area + 1e-16)
    # print(b2_x1, b2_y1, b2_x2, b2_y2, bbox2[:, 0], bbox2[:, 2], bbox2[:, 1], bbox2[:, 3])
    # print("sd", b1_x1, b1_y1, b1_x2, b1_y2, bbox1[:, 0], bbox1[:, 2], bbox1[:, 1], bbox1[:, 3])
    # print(area, b1_area, b2_area,iou)
    # exit()
    return iou

def boxes_change(prediction, imgShape, target):
    # 不用变换图像格式了前面变过了
    prediction = prediction.view(-1,7)
    # 进行prediction的框的变换，变换到原来图片的尺度之下
    height, width = imgShape
    target_h, target_w = target
    scale_h, scale_w = target_h / height, target_w / width
    minScale = min(scale_h, scale_w)
    after_w, after_h = int(width*minScale), int(height * minScale)
    pad_w = target_w - after_w
    pad_h = target_h - after_h
    # print(prediction[..., [0,2]], pad_w/2)
    # print(prediction[..., [1,3]], pad_h/2)
    #  S: 这里的减法和除法有问题
    minus = torch.Tensor([pad_w/2, pad_h/2, pad_w/2, pad_h/2]).repeat(prediction.size(0),1)
    # print(boxes)
    prediction[..., 0:4] -= minus
    # prediction[..., [0,2]] -= pad_w/2
    # prediction[..., [1,3]] -= pad_h/2
    # print(boxes)
    prediction[..., 0:4] /= minScale
    # print(boxes,"\n///")

    # 将超过图片范围的框进行剪裁 到图片内
    # print(height, width)
    prediction[..., [0, 2]] = torch.clamp(prediction[..., [0, 2]], 0.0, width)
    prediction[..., [1, 3]] = torch.clamp(prediction[..., [1, 3]], 0.0, height)
    # print(prediction)
    return prediction