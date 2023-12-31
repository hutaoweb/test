print("work start!")
import torch

print(torch.__version__)
# import tensorboard_logger as tb_logger
print("import logger OK!")
from model import EncoderImage,EncoderText,BuildTrainNetwork,BuildValNetwork, CustomTrainOneStepCell,SimLoss
import numpy as np
from collections import OrderedDict
import yaml
from easydict import EasyDict
from ipdb import set_trace

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mindspore import load_checkpoint, load_param_into_net
import logging
import torch.backends.cudnn as cudnn
import pickle
import os
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

import data
from model import ImageSelfAttention
import model
from vocab import Vocabulary
import argparse
from fusion_module import *
import mindapore as ms


def test_CAMP_model(config_path):
    print("OK!")
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    # config_path = "./experiments/f30k_cross_attention/config_test.yaml"
    with open(config_path) as f:
        opt = yaml.safe_load(f)  # ,Loader=yaml.Loader
    opt = EasyDict(opt['common'])

    # set_trace()

    with open(os.path.join(opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    # vocab = pickle.load(open(os.path.join(opt.vocab_path,'%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)

    train_logger = LogCollector()

    print("----Start init model----")
    img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                opt.finetune, opt.cnn_type,
                                no_imgnorm=opt.no_imgnorm,
                                self_attention=opt.self_attention)

    txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                               opt.embed_size, opt.num_layers,
                               no_txtnorm=opt.no_txtnorm,
                               self_attention=opt.self_attention,
                               embed_weights=opt.word_embed,
                               bi_gru=opt.bi_gru)

    #定义损失模型
    criterion = SimLoss(margin=opt.margin,
                             measure=opt.measure,
                             max_violation=opt.max_violation,
                             inner_dim=opt.embed_size)
    testnet = BuildValNetwork(img_enc, txt_enc, criterion)

    # 加载模型
    if opt.resume:
        state = ms.load_checkpoint(opt.resume)  #state是字典：{'name': name, 'data': param}
        #创建图像、文本、损失字典
        checkpoint_dict = {"net_image":{},"net_caption":{},"criterion":{}}
        for key, value in state.items():
#             set_trace()
            qianzu = key.split(".")[0]
            checkpoint_dict[qianzu][key] = value
        #加载
        ms.load_param_into_net(img_enc, checkpoint_dict["net_image"])
        ms.load_param_into_net(txt_enc, checkpoint_dict["net_caption"])
        ms.load_param_into_net(criterion, checkpoint_dict["criterion"])

    testnet.set_train(False)
    print("加载模型完成")

    print("开始加载数据")
    test_loader, test_loader_len = data.get_test_loader("test", opt.data_name, vocab, opt.crop_size, 128, 4, opt)

    print("开始前向传播计算数据特征")
    img_embs, cap_embs, cap_masks = encode_data(
        CAMP, test_loader, opt.log_step, logging.info)

    print("开始计算指标")
    (r1, r5, r10, medr, meanr), (r1i, r5i, r10i, medri, meanri), score_matrix = i2t(img_embs, cap_embs, cap_masks,
                                                                                    measure=opt.measure,
                                                                                    model=CAMP, return_ranks=True)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))


def main():
    # test_f30k_dataloader()
    # test_text_encoder()
    # test_img_encoder()
    # test_stack_fusion()
    # test_gate_fusion()
    # test_stack_fusion_new()
    # test_gate_fusion_new()
    test_CAMP_model("CAMP/test.yaml")


if __name__ == '__main__':
    main()
