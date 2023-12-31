import argparse
import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils import data, model_zoo
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from ipdb import set_trace
from mindspore import load_checkpoint, load_param_into_net
# from model_mindspore.deeplab_multi import DeeplabMulti
# from model import  EncoderImagePrecomp,EncoderText
from mindspore import save_checkpoint, Tensor
import torch

"""
定义模型
"""
class EncoderImagePrecomp(nn.Cell):

    def __init__(self, img_dim, embed_size,no_imgnorm):
        super(EncoderImagePrecomp, self).__init__()
        img_dim = 2048
        embed_size = 1024
        self.fc = nn.Dense(img_dim, embed_size)



    def construct(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        return features


"""
输入torch模型的参数名称
输出列表：元素是二元组（变换之后权重名称，原本的权重名称）
"""
def update_name(list_old):
    # list = ['.'.join(name.split('.')[1:]) for name in list_old]
    list = [name for name in list_old]
    list_ = []
    for name, name_old in zip(list, list_old):
        # pre_name = name.split('.')[0]
        # if pre_name != 'layer5' and pre_name != 'layer6':
        list_.append((name, name_old))
    return list_

"""
将经过update_name函数变换之后的权重名称，进行转化成ms的权重名称
"""
def name_raplace(name):
    if "embed.weight" in name:
        name = "net_caption.embed.embedding_table"
    else:
        # name = "net_caption." + name
        name = "net_image." + name
    return name
    # torch 和mindspore部分参数名不一致，进行修改，以下修改bn层参数名
    # bn_dict = {'weight': 'gamma', 'bias': 'beta',
    #            'running_mean': 'moving_mean', 'running_var': 'moving_variance'}
    # old = name.split('.')[-1]
    # return name
    # if 'bn' in name:
    #     return name.replace(old, bn_dict[old])
    # elif ('downsample' in name) and (name.split('.')[-2] != '0'):
    #     # print(name)
    #     return name.replace(old, bn_dict[old])
    # else:


        # return name


def updata_torch_to_ms(static_dict_ms, static_dict_torch):
    # new_static_dict=dict()
    key_list = update_name(static_dict_torch.keys()) # 我的预训练模型需要舍去resnet后两层，所以有这个步骤
    for key, key_old in key_list:
        key_ = name_raplace(key) # 置换参数名
        print("----------------"+ key_ +"------------------")   
        param = static_dict_torch[key_old]
        new_param = mindspore.Tensor(param.numpy())
        print('目标torch参数：', param)
        print('转换ms参数：', new_param)
        # print(key,'\t||||\t',key_)
        print(key_)
        static_dict_ms[key_].set_data(new_param) # 加载新权重
    return static_dict_ms

#加载torch权重
model_path = "C:/Users/LiaoYu/Desktop/SCAN_ms/torch_to_ms/run_flicker30/model_best.pth.tar"
checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
static_dict1 = checkpoint['model'][0]  #0图模型权重， 1为文本模型权重
# static_dict1 = torch.load(r'D:\Files\GitHub\AdvSeg-Mindspore\model\DeepLab_resnet_pretrained_init-f81d91e8.pth')
# print(saved_static_dict)
# model = DeeplabMulti()
# print('原始参数：', list(model.parameters_dict().values())[0][0][0][0])
# mindspore.save_checkpoint(model,r'model_mindspore/DeeplabMulti.ckpt')
# # print(model.parameters_dict().keys())
# static_dict2=mindspore.load_checkpoint(r'model_mindspore/DeeplabMulti.ckpt')
#加载ms模型权重
# model_path = "/data1/ly/huawei/SCAN_ms/runs/run_flicker30/checkpoint/model/"
image_weight_path = "C:/Users/LiaoYu/Desktop/SCAN_ms/torch_to_ms/run_flicker30/imagemodel_best.ckpt"
# text_weight_path = model_path + "text" + 'model_best.ckpt'
# net_ms_param_dict = load_checkpoint(image_weight_path)
# net_ms = EncoderImagePrecomp(2048, 1024, 1)
# load_param_into_net(net_ms, net_ms_param_dict)
# static_dict2 = net_ms.parameters_dict()
static_dict2 = load_checkpoint(image_weight_path)

# print(value_list2[1].asnumpy(),value_list2[2].shape,value_list2[3].shape,value_list2[4].shape)
static_dict_ms = updata_torch_to_ms(static_dict2, static_dict1)

value_list2 = list(static_dict_ms.values())
ms_name_list = list(static_dict_ms.keys())
final_torch_to_ms_weight = []
for i in range(len(value_list2)):  
    final_torch_to_ms_weight.append({"name": ms_name_list[i], "data": value_list2[i]})
save_checkpoint(final_torch_to_ms_weight, "imageweight_torch_to_ms.ckpt")

# net_ms_param_dict = load_checkpoint(image_weight_path)
# print(net_ms_param_dict)


