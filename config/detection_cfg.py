DETECTION_PARAMS = {
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "batch_size": 1,
    "iou_thres": 0.5,
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    "pretrain_snapshot": "./weights/official_yolov3_weights_pytorch.pth",
    "input_path": "./data/test/",
    "output_path": "./output/",
    "use_cuda": True,
    "classes_name_path": "./data/coco.names"
}

"""
    model_params:{
        backbone_name: 用来选择是 darknet_53 还是 darknet_21的
        backbone_pretrained: 用来指定backbone 的预训练参数的文件路径的
    },
    yolo:{
        anchors: 默认的anchor检测框
    },
    batch_size: batch大小
    iou_thres: nms用到的阈值大小
    img_h: 输入高
    img_w: 输入宽
    parrallels: 可并行gpu
    pretrain_snapshot: 预训练模型路径(这个是整体的模型的预训练数据)
    input_path: 记录所有要测试的图片的路径
    output_path: 记录了图片测试检测之后输出保存的路径
    use_cuda: 是否使用cuda
    classes_name_path: 存储检测类别的文件路径

"""