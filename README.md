## 项目目录结构
```Python
---MyYoloV3
    ---config 配置文件
        train_cfg.py 训练参数文件
        detection_cfg.py 检验参数文件
    ---checkpoints 检查点文件
        #格式: /开始时间/batch_{i}/
    ---model 模型文件夹
        Darknet.py #backbone模块文件
        YoloNet.py #YoloNet模块文件
    ---data 数据文件夹
        ---train 包含label的txt文件和图片文件
        ---test
        train.txt #训练的文件列表
    ---output 输出图片文件夹
    ---weight 网络权重文件夹
    ---util 工具文件
        util.py 工具文件
        PictureDataLoader.py DataLoader图片加载器实现
    train.py 训练文件
    detection.py 检验文件
```