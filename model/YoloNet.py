import torch
import torch.nn as nn
import numpy as np
from util.modelChoose import backboneChoose

class YoloNet(nn.Module):
    def __init__(self, config, isTraining = True):
        super(YoloNet, self).__init__()
        self.config = config
        self.isTraining = isTraining
        # 构建backbone
        self.module_params = config["model_params"]
        backbone_ = backboneChoose[self.module_params["backbone_name"]]
        self.backbone = backbone_(self.module_params["backbone_pretrained"])
