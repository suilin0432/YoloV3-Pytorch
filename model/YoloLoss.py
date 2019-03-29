import torch
import torch.nn as nn
import numpy as np
from util.util import bbox_iou

class YoloLoss(nn.Module):
    def __init__(self, anchors, imgShape, classes, ignoreThreshold=0.5):
        super(YoloLoss, self).__init__()
        self.anchors = anchors
        self.imgShape = imgShape
        self.numAnchors = len(anchors)
        self.ignoreThreshold = ignoreThreshold
        self.classes = classes

        # 设定损失函数各部分的权重
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_class = 1.0
        self.lambda_confidence = 1.0

        # 损失函数用到的类别
        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCELoss()

    def forward(self, prediction, targets):
        # 进行损失函数的计算以及返回最后的整体的损失函数
        # 首先要将prediction进行处理
        # 先将batchsize, anchorNums*(5+classesNum), height, width转化为
        #    batchSize,anchorNums, height, width, 5 + classesNum
        stride = self.imgShape[0] / prediction.size(2)
        prediction = prediction.view(prediction.size(0), self.numAnchors, self.classes + 5, prediction.size(2), prediction.size(3))
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]

        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 获取与GT对应的相关信息进行Loss 的计算
        mask, noobjMask, tx, ty, tw, th, tconf, tcls = self.GTCalculate(targets, scaled_anchors, prediction.size(2), prediction.size(3), self.ignoreThreshold)
        if torch.cuda.is_available():
            mask = mask.cuda()
            noobjMask = noobjMask.cuda()
            tx, ty, tw, th, tconf, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), tconf.cuda(), tcls.cuda()
        loss_x = self.BCELoss(x * mask, tx * mask)
        loss_y = self.BCELoss(y * mask, ty * mask)
        loss_w = self.MSELoss(w * mask, tw * mask)
        loss_h = self.MSELoss(h * mask, th * mask)
        loss_conf = self.BCELoss(conf * mask, mask) + 0.5 * self.BCELoss(conf * noobjMask, noobjMask * 0.0)
        loss_cls = self.BCELoss(pred_cls[mask == 1], tcls[mask == 1])
        loss = self.lambda_xy * (loss_x + loss_y) + self.lambda_wh * (loss_w + loss_h) + self.lambda_class * loss_cls + self.lambda_confidence * loss_conf
        return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item()

    def GTCalculate(self, targets, scaled_anchors, featureWidth, featureHeight, ignoreThreshold):
        batchSize = targets.size(0)
        mask = torch.zeros(batchSize, self.numAnchors, featureHeight, featureWidth, requires_grad=False)
        noobjMask = torch.ones(batchSize, self.numAnchors, featureHeight, featureWidth, requires_grad=False)
        tx = torch.zeros(batchSize, self.numAnchors, featureHeight, featureWidth, requires_grad=False)
        ty = torch.zeros(batchSize, self.numAnchors, featureHeight, featureWidth, requires_grad=False)
        tw = torch.zeros(batchSize, self.numAnchors, featureHeight, featureWidth, requires_grad=False)
        th = torch.zeros(batchSize, self.numAnchors, featureHeight, featureWidth, requires_grad=False)
        tconf = torch.zeros(batchSize, self.numAnchors, featureHeight, featureWidth, requires_grad=False)
        tcls = torch.zeros(batchSize, self.numAnchors, featureHeight, featureWidth, self.classes, requires_grad=False)

        # 开始进行每个batch的遍历
        for batch in range(batchSize):
            # 开始进行每个detection的遍历
            for bbox in targets[batch]:
                if targets.sum() == 0:
                    #表示是为了填补到 规定的anchor尺寸而进行的0补 跳过即可
                    continue
                #进行GT的尺度变换
                boxX = bbox[1] * featureWidth
                boxY = bbox[2] * featureHeight
                boxW = bbox[3] * featureWidth
                boxH = bbox[4] * featureHeight

                #获取 中心点坐标
                centerX, centerY = int(boxX), int(boxY)

                #开始计算GT框和中心点对应的Iou
                GT_ = torch.Tensor([0,0, boxW, boxH])
                GT_ = GT_.unsqueeze(0)
                ANCHOR_ = torch.cat((torch.zeros(self.numAnchors, 2), torch.Tensor(scaled_anchors)), 1)

                anchorIoU = bbox_iou(GT_, ANCHOR_)

                # 对所有IoU >threshold 的noobjMask赋值为0
                noobjMask[batch, anchorIoU > 0.5, centerY, centerX] = 0

                # 然后找出最好的那个框
                bestIndex = torch.argmax(anchorIoU)
                mask[batch, bestIndex, centerY, centerX] = 1

                tx[batch, bestIndex, centerY, centerX] = boxX - centerX
                ty[batch, bestIndex, centerY, centerX] = boxY - centerY
                tw[batch, bestIndex, centerY, centerX] = torch.log(boxW / scaled_anchors[bestIndex][0] + 1e-16)
                th[batch, bestIndex, centerY, centerX] = torch.log(boxH / scaled_anchors[bestIndex][1] + 1e-16)
                tconf[batch, bestIndex, centerY, centerX] = 1
                tcls[batch, bestIndex, centerY, centerX, int(bbox[0])] = 1
        return mask, noobjMask, tx, ty, tw, th, tconf, tcls


