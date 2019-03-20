import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2

class PictureDataLoader(DataLoader):
    def __init__(self, imgRecordFile):
        """

        :param imgRecordFile: 记录 文件目录的文件
        """
        # 读取记录所有图片的路径
        self.imgRecordFile = imgRecordFile
        imgSrcListFile = open(imgRecordFile, 'r')
        self.imgSrcList = imgSrcListFile.readlines()
        self.imgSrcList =


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass