import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
# 引入配置文件
from config.detection_cfg import DETECTION_PARAMS as cfg
from model.YoloNet import YoloNet
from util.PictureDataLoader import PictureDataLoader
import os
import logging
import cv2
import pickle as pkl
from util.util import detection_anchors, detection_img_input_convert, boxes_change
import random

def load_classes(cocoNames):
    fp = open(cocoNames, "r")
    # 最后一个是一个空行, 所以要去掉最后一行
    names = fp.read().split("\n")[:-1]
    return names

if cfg["img_h"] % 32 != 0 or cfg["img_w"] % 32 != 0:
    raise Exception("please set the width and height the numbers which can be divided by 32")

# 一些参数
CUDA = torch.cuda.is_available() and cfg["use_cuda"]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg["parallels"]))
logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s %(filename)s] %(message)s")
targetSize = (cfg["img_h"], cfg["img_w"])
classes = load_classes(cfg["classes_name_path"])
anchors = cfg["yolo"]["anchors"]
anchors = torch.Tensor(anchors)
colors = pkl.load(open("pallete", "rb"))
# 加载网络
net = YoloNet(config=cfg, isTraining=True)
is_training = False
net.train(is_training)
if CUDA:
    net = nn.DataParallel(net)
    net = net.cuda()

# 加载权重
pretrain_snapshot_path = cfg["pretrain_snapshot"]
if pretrain_snapshot_path:
    logging.info("loading the yolov3 weights from {0} !".format(pretrain_snapshot_path))
    state_dict = torch.load(pretrain_snapshot_path)
    net.load_state_dict(state_dict)
else:
    raise Exception("No yolov3 weights found in {0} !".format(pretrain_snapshot_path))

# 加载图片列表路径
imgNameList = os.listdir(cfg["input_path"])
imgPathList = [os.path.join(cfg["input_path"], i) for i in imgNameList]
if len(imgPathList) == 0:
    raise Exception("no image found in {}".format(cfg["input_path"]))

for imgPath in imgPathList:
    img = cv2.imread(imgPath)
    originImg = img.copy()
    imgShape = img.shape
    # 进行图像的变换
    img = detection_img_input_convert(img, targetSize)
    if CUDA:
        img = img.cuda()

    # 因为只取出来了一张图片所以要加上第0维
    img = img.unsqueeze(0)

    # 图片索引
    index = 0
    # 进行prediction的计算获取
    with torch.no_grad():
        prediction = net(img)

        # 进行prediction的处理 (包括变换, NMS)

        prediction = detection_anchors(prediction, CUDA,
                                       [cfg["img_h"]/32, cfg["img_h"]/16, cfg["img_h"]/8],
                                       anchors)
        # 进行框的图片框的绘制
        # PS: 虽然在util里面做的是可以批处理的, 但是实际上一般检测就是一张而已
    for i in range(1):
        pre = prediction[i]
        # 进行边界框的变换(因为边界框现在是基于缩放后的图片的所以...), 以及转变为左上角 右下角的格式
        pre = boxes_change(pre, imgShape, targetSize)
        for boxes in pre:
            #开始进行绘制
            upLeft = tuple(boxes[1:3].int())
            rightDown = tuple(boxes[3:5].int())
            cls = boxes[-1]
            label = "{}".format(classes[cls])
            color = random.choice(colors)
            cv2.rectangle(originImg, upLeft, rightDown, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = upLeft[0] + t_size[0] + 3, upLeft[1] + t_size[1] + 4
            cv2.rectangle(img, upLeft, c2, color, -1)
            cv2.putText(img, label, (upLeft[0], c2[1]), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
        cv2.imwrite(os.path.join(cfg["output_path"],index), originImg)
        index += 1



