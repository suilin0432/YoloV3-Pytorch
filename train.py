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


CUDA = torch.cuda.is_available() and cfg["use_cuda"]