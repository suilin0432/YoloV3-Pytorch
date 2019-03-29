TRAIN_PARAMS = {
    "model_params":{
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo":{
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "batch_size": 4,
    "iou_thres": 0.5,
    "img_h": 416,
    "img_w": 416,
    "parallels": [0, 1],
    "pretrain_snapshot": "",
    "input_path": "./data/train.txt",
    "output_path": "./checkpoints/",
    "use_cuda": True,
    "epoch": 100,
    "lr":{
        "backbone_lr": 0.001,
        "other_lr": 0.01,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 20,           #  decay lr in every ? epochs
    },
    "optimizer":{
        "type": "sgd",
        "weight_decay": 4e-05,
    }
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
    input_path: 记录所有训练数据的文件的路径
    output_path: 记录了训练权重保存的路径
    use_cuda: 是否使用cuda
    epoch: 训练的epoch数目
    
"""