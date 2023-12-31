# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Training script"""
# import torch

import os
import time
import shutil
import sys
# import torch
import numpy
import numpy as np

import data
from vocab import Vocabulary, deserialize_vocab
from mindspore import context
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn_t2i #, shard_xattn_i2t
from mindspore import load_checkpoint, load_param_into_net
from tqdm import tqdm
import logging
import tensorboard_logger as tb_logger

import argparse
from ipdb import set_trace
import json
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from time import *
import copy

from model import save_state_dict
from model import ContrastiveLoss,EncoderImage,EncoderText,BuildTrainNetwork,BuildValNetwork, CustomTrainOneStepCell








def main():
    
    # Hyper Parameters
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/',   #   ./data/
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp',                                      #precomp
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./data/vocab/', #./vocab/
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=20, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,                              
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=10, type=int,                                       #15
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/run_flicker30/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/run_flicker30/checkpoint',      #./runs/runX/checkpoint
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--agg_func', default="LogSumExp",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--cross_attn', default="t2i",
                        help='t2i|i2t')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--bi_gru', action='store_true',
                        help='Use bidirectional GRU.')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--device_target', default="Ascend", type=str,
                        help='Attention softmax temperature.')
    # parser.add_argument('--device_target', default=2, type=int,
    #                     help='Attention softmax temperature.')
    opt = parser.parse_args()

    
    if not os.path.exists(opt.model_name):
        os.mkdir(opt.model_name)
    argsDict = opt.__dict__
    print(argsDict)
    js = json.dumps(argsDict,indent = 4)
    parameter_path = opt.model_name+ "/" + "config.json"
    file = open(parameter_path, 'w', encoding='utf-8')
    file.write(js + '\n')
    file.close()
    # set_trace()

    #context.PYNATIVE_MODE
    #context.GRAPH_MODE
#     context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)
    train_dataset, val_loader,train_dataset_len,val_dataset_len = data.get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)
    print("train nums:",train_dataset_len)
    print("test nums", val_dataset_len)



    # model
    txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                               opt.embed_size, opt.num_layers,
                               use_bi_gru=opt.bi_gru,
                               no_txtnorm=opt.no_txtnorm,
                              batch_size=opt.batch_size)
    img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                            no_imgnorm=opt.no_imgnorm)

    # loss model
    criterion = ContrastiveLoss(lambda_softmax = opt.lambda_softmax, 
                                     agg_func = opt.agg_func, 
                                     lambda_lse = opt.lambda_lse, 
                                     cross_attn = opt.cross_attn,
                                     raw_feature_norm = opt.raw_feature_norm,
                                     margin = opt.margin,
                                     max_violation = opt.max_violation)
    net_with_loss = BuildTrainNetwork(img_enc,txt_enc, criterion)
    valnet = BuildValNetwork(img_enc,txt_enc, criterion)


    #loss model  opt
    batch_each_epoch = (train_dataset_len // 128) + 1
    milestone = []
    learning_rates = []
    for i in range(opt.num_epochs):
        milestone.append((i+1)*batch_each_epoch)
        learning_rates.append(opt.learning_rate * (0.5 ** (i // opt.lr_update)))
    #set_trace()
    output = nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)
    params = list(txt_enc.trainable_params())
    # set_trace()
    params += list(img_enc.fc.trainable_params())
    # set_trace()
    # self.optimizer = nn.Adam(self.txt_enc.trainable_params(), learning_rate=output)
    optimizer = nn.Adam(params, learning_rate=output)  #output
    train_net = CustomTrainOneStepCell(net_with_loss, optimizer)
    #self.model = nn.TrainOneStepCell(self.trainloss, opt)
    squeeze = ops.Squeeze()

#     #加载模型的权重
#     model_path = "temp/"
#     image_weight_path = model_path + "image" + 'model_best.ckpt'
#     text_weight_path = model_path + "text" + 'model_best.ckpt'
# #     train_net.load_state_dict(image_weight_path, text_weight_path)
#     image_param_dict = load_checkpoint(image_weight_path)
#     load_param_into_net(img_enc, image_param_dict)
#     text_param_dict = load_checkpoint(text_weight_path)
#     load_param_into_net(txt_enc, text_param_dict)



    print("train start")
    steps = train_dataset.get_dataset_size()
    best_rsum = 0
    # 设置网络为训练模式
    train_net.set_train()
    begin_time = time()
    for epoch in range(opt.num_epochs):
        step = 0
        print("-----------------   "+ "epoch " + str(epoch+1) + "   -------------------")
# #         for i, (images,  captions, lengths, ids) in enumerate(train_loader):
        for d in train_dataset.create_dict_iterator():
#             set_trace()
            lengths = squeeze(d["lengths"])
            lengths_int = lengths.asnumpy().tolist()
#             set_trace()
            result = train_net(d["images"],d["captions"], lengths, d["caption_mask"])
            print(f"Epoch: [{epoch+1} / {opt.num_epochs}], "
              f"step: [{step} / {steps}], "
                 f"loss: {result}, ")
            step = step + 1
        prefix = opt.model_name + '/'
        save_state_dict(img_enc, txt_enc, prefix, epoch, is_best=False)
    end_time = time()
    run_time = end_time-begin_time
    print ('time: ',run_time)
    # 验证
    
#         rsum = validate(opt, val_loader, valnet, val_dataset_len)

# #         remember best R@ sum and save checkpoint
#         is_best = rsum > best_rsum
#         best_rsum = max(rsum, best_rsum)


#     prefix = opt.model_name + '/'
#     save_state_dict(img_enc, txt_enc, prefix, epoch, is_best=False)
#         if is_best:
#             save_state_dict(img_enc, txt_enc, prefix, epoch, is_best=is_best)
#             model.save_state_dict(prefix, epoch, is_best=is_best)



def validate(opt, val_loader, model,val_dataset_len):
    model.set_train(False)
    # 
    img_embs, cap_embs, caption_masks, cap_lens = encode_data(
        model, val_loader, opt.log_step, logging.info,val_dataset_len)

    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time()
#     set_trace()
    if opt.cross_attn == 't2i':   
        # set_trace()
        sims = shard_xattn_t2i(img_embs, 
                               cap_embs, 
                               caption_masks, 
                               opt, 
                               shard_size=100, 
                               caplens=cap_lens)
    # elif opt.cross_attn == 'i2t':
    #     sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    # else:
    #     raise NotImplementedError
    # set_trace()
    end = time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %(r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, cap_lens, sims)
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %(r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    print(currscore)
    return currscore




# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


if __name__ == '__main__':
    main()
