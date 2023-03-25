import os

cur_dir = "./"
miniimagenet_split_path = "./split"

# 此处存放MiniImagenet, 文件夹下是60000张图片，格式大概为n0153282900000005.jpg
miniimagenet_path = r"C:\Workspace\work\datasets\MiniImagenet"


pretrain_fpaths = {
    "ConvNet": "./pretrain/miniimagenet/con-pre.pth",
    "ResNet": "./pretrain/miniimagenet/Res12-pre.pth",
}
