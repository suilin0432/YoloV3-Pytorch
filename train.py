from config.train_cfg import TRAIN_PARAMS as cfg
import torch
import torch.nn as nn
import numpy
from model.YoloNet import YoloNet
from model.YoloLoss import YoloLoss
from torch.utils.data import DataLoader
from util.PictureDataLoader import PictureDataLoader
import os
import logging
import cv2
import torch.optim as optim
import time

# 别处抄来的函数
def get_optimizer(config, net):
    optimizer = None
    params = None
    #
    base_params = list(
        map(id, net.backbone.parameters())
    )
    # id() 是获取对象的地址的函数
    # 这个是把base_params之外的变量筛选出来，留下来backbone之外的参数
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    # 开始构建 params变量
    if not config["lr"]["freeze_backbone"]:
    # 如果 没有设置freeze_backbone的话 那么将backbone 和 后加的参数进行不同的学习率进行学习
    # 设置中backbone的参数会比其他参数学习率要低
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    # 如果设置freeze_backbone的话就以相同的学习率进行训练
    else:
        logging.info("freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
        ]
    # 根据所设置的不同类型的优化器方法进行optimizer的构建
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        # Default to sgd
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer

def _save_checkpoint(state_dict, config, evaluate_func=None):
    # global best_eval_result
    checkpoint_path = os.path.join(config["sub_working_dir"], "model.pth")
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)


# 一些参数设置
if cfg["img_h"] % 32 != 0 or cfg["img_w"] % 32 != 0:
    raise Exception("please set the width and height the numbers which can be divided by 32")
CUDA = torch.cuda.is_available() and cfg["use_cuda"]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cfg["parallels"])
logging.basicConfig(level=logging.DEBUG,format="[%(asctime)s %(filename)s] %(message)s")
anchors = cfg["yolo"]["anchors"]
imgShape = (cfg["img_h"], cfg["img_w"])
classes = cfg["yolo"]["classes"]
ignoreThreshold = 0.5
epoches = cfg["epoch"]
backbone_pretrained = cfg["model_params"]["backbone_pretrained"]
pretrain_snapshot = cfg["pretrain_snapshot"]
config["global_step"] = config.get("start_step", 0)
# DataLoader
dataloader = DataLoader(
    PictureDataLoader(cfg["input_path"], (416, 416)),
    batch_size=cfg["batch_size"],
    num_workers=32,
)

# 网络的初始化
net = YoloNet(cfg, True)
net.train(True)

# optimizer初始化
optimizer = get_optimizer(cfg, net)
# 设置什么时候进行学习率的衰减
lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["lr"]["decay_step"],
        gamma=cfg["lr"]["decay_gamma"])

# 将网络并行
if CUDA:
    net = nn.DataParallel(net)
    net = net.cuda()

if pretrain_snapshot:
    logging.info("Load pretrained weights from {}".format(cfg["pretrain_snapshot"]))
    state_dict = torch.load(cfg["pretrain_snapshot"])
    net.load_state_dict(state_dict)

# 设置yolo_loss
yolo_losses = []
for i in range(3):
    yolo_losses.append(YoloLoss(cfg["yolo"]["anchors"][i], cfg["yolo"]["classes"], (cfg["img_w"], cfg["img_h"]), ignoreThreshold))

# 开始进行 训练
logging.log("begin training!")
for epoch in range(epoches):
    for step, samples in enumerate(dataloader):
        # 读取图片和对应的标签
        images, labels = samples["img"], samples["label"]
        start_time = time.time()
        # 记录当前执行到的epoch数目 +1 为了断点进行的设置
        cfg["global_step"] += 1

        # 进行前向传播然后反向传播更新参数设置
        optimizer.zero_grad()
        outputs = net(images)
        losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
        losses = []
        for _ in range(len(losses_name)):
            losses.append([])
        for i in range(3):
            _loss_item = yolo_losses[i](outputs[i], labels)
            for j, l in enumerate(_loss_item):
                losses[j].append(l)
        losses = [sum(l) for l in losses]
        loss = losses[0]
        loss.backward()
        optimizer.step()

        if step > 0 and step % 10 == 0:
            _loss = loss.item()
            duration = float(time.time() - start_time)
            example_per_second = cfg["batch_size"] / duration
            lr = optimizer.param_groups[0]['lr']
            logging.info(
                "epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f " %
                (epoch, step, _loss, example_per_second, lr)
            )
            for i, name in enumerate(losses_name):
                value = _loss if i == 0 else losses[i]

        if step > 0 and step % 1000 == 0:
            # net.train(False)
            _save_checkpoint(net.state_dict(), cfg)
            # net.train(True)

    lr_scheduler.step()

    # net.train(False)
    _save_checkpoint(net.state_dict(), cfg)
    # net.train(True)
    logging.info("Bye~")

