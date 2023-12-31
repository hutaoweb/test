# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Evaluation"""

# import torch

# from __future__ import print_function
import os
from mindspore import load_checkpoint, load_param_into_net
import sys
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary, deserialize_vocab  # NOQA
# import torch
from model import xattn_score_t2i_xin#, xattn_score_i2t
from collections import OrderedDict
import time
# from torch.autograd import Variable
from ipdb import set_trace
import copy
from mindspore import ops
import argparse
import json
from mindspore import nn
import mindspore as ms
from model import ContrastiveLoss,EncoderImage,EncoderText,BuildTrainNetwork,BuildValNetwork, CustomTrainOneStepCell

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    # X:  (128, 36, 1024)
    # dim: -1
#     set_trace()
    norm = ops.Sqrt()(ops.Pow()(X, 2).sum(axis=dim, keepdims=True)) + eps   #(128, 36, 1)
    X = ops.Div()(X, norm)   #(128, 36, 1024)
    # norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    # X = torch.div(X, norm)
    # set_trace()
    return X

def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)   是扩展的文本
    context: (n_context, sourceL, d)  是一个batch的图像
    """
    batch_size_q, queryL = query.shape[0], query.shape[1]
    batch_size, sourceL = context.shape[0], context.shape[1]


    # Get attention
    # --> (batch, d, queryL)
    queryT = ops.Transpose()(query, (0, 2, 1))
    # queryT = query.view((query.shape[0], query.shape[2],query.shape[1]))

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    #两个batch后的Tensor之间的矩阵乘法。
    # attn = torch.bmm(context, queryT)
    context = context
    # set_trace()
    attn = ops.BatchMatMul()(context, queryT)   #(128, 36, n)
    # set_trace()
    if raw_feature_norm == "clipped_l2norm":   #执行这个
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    # elif opt.raw_feature_norm == "softmax":
    #     # --> (batch*sourceL, queryL)
    #     attn = attn.view(batch_size*sourceL, queryL)
    #     attn = nn.Softmax()(attn)
    #     # --> (batch, sourceL, queryL)
    #     attn = attn.view(batch_size, sourceL, queryL)
    # elif opt.raw_feature_norm == "l2norm":
    #     attn = l2norm(attn, 2)
    # elif opt.raw_feature_norm == "l1norm":
    #     attn = l1norm_d(attn, 2)
    # elif opt.raw_feature_norm == "clipped_l1norm":
    #     attn = nn.LeakyReLU(0.1)(attn)
    #     attn = l1norm_d(attn, 2)
    # elif opt.raw_feature_norm == "clipped":
    #     attn = nn.LeakyReLU(0.1)(attn)
    # elif opt.raw_feature_norm == "no_norm":
    #     pass
    # else:
    #     raise ValueError("unknown first norm type:", opt.raw_feature_norm)

    # --> (batch, queryL, sourceL)
    # attn = torch.transpose(attn, 1, 2).contiguous()
    # set_trace()
    attn = ops.Transpose()(attn, (0, 2, 1))   ##(128, n, 36)
    # --> (batch*queryLn, sourceL36)
    # set_trace()
    attn = attn.view(batch_size*queryL, sourceL)   #(128*n, 36)
    # set_trace()
    attn = nn.Softmax()(attn*smooth)
    # set_trace()
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)   #(128, n, 36)
    # set_trace()
    # --> (batch, sourceL, queryL)
    attnT = ops.Transpose()(attn, (0, 2, 1))#.contiguous()  #(128, 36, n)

    # --> (batch, d, sourceL)
    contextT = ops.Transpose()(context, (0, 2, 1))   #(128, 1024, 36)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = ops.BatchMatMul()(contextT, attnT)   #(128, 1024, 12)
    # --> (batch, queryL, d)
    # set_trace()
    weightedContext = ops.Transpose()(weightedContext, (0, 2, 1))  #(128, 12, 1024)
    # set_trace()
    return weightedContext, attnT  #(128, 12, 1024)    (128, 36, n)

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    #x1,x2 (8, 20, 1024)
    """Returns cosine similarity between x1 and x2, computed along dim."""
    # w12 = torch.sum(x1 * x2, dim)
    # w12 = ops.ReduceSum()(x1 * x2,dim)   #8,20
    # # #计算x1和x2的模值   对应位置相乘、相加、开根号
    # w1 = x1 * x1
    # w2 = x2 * x2
    # # set_trace()
    # w1 = ops.Sqrt()(ops.ReduceSum()(w1, -1))  #8,20
    # w2 = ops.Sqrt()(ops.ReduceSum()(w2, -1))  #8,20
    # # set_trace()
    # temp = w12 / ops.clip_by_value(w1 * w2, 0.000001, (w1 * w2).max())
    # set_trace()
    # set_trace()
    # w1 = torch.norm(x1, 2, dim)
    # w1 = ops.LpNorm(axis=dim, p=2)(x1)
    # w2 = torch.norm(x2, 2, dim)
    # w2 = ops.LpNorm(axis=dim, p=2)(x2)
    # return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    # temp = w12 / (w1 * w2).clamp(min=eps)
    # eps = ms.Tensor(eps, ms.float32)
    # temp = w12 / ops.clip_by_value(w1 * w2,eps)



    x1 = ops.L2Normalize(axis=-1, epsilon=1e-4)(x1)
    x2 = ops.L2Normalize(axis=-1, epsilon=1e-4)(x2)
    temp = ops.ReduceSum()(x1 * x2,dim)
    # set_trace()
    return ops.Squeeze()(temp) #??????
#
def xattn_score_t2i(images, captions, cap_lens,
                    lambda_softmax,
                    agg_func,
                    lambda_lse,
                    raw_feature_norm):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.shape[0] #128
    n_caption = captions.shape[0]  #128
    # set_trace()
    for i in range(n_caption):
        # Get the i-th text description
        n_word = int(cap_lens[i])

        cap_i = ms.ops.ExpandDims()(captions[i, :n_word, :], 0)   #(1, 15, 1024)
        # cap_i = .contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = ms.numpy.tile(cap_i,(n_image, 1, 1))  #(128, 15, 1024)
        # set_trace()
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        # set_trace()
        weiContext, attn = func_attention(cap_i_expand, images, raw_feature_norm, smooth=lambda_softmax)   ##(128, 12, 1024)    (128, 36, n)
        # cap_i_expand = cap_i_expand#.contiguous()   (b, n, 1024)   (8, 15, 1024)
        # weiContext = weiContext#.contiguous()       (b, n, 1024)
        # (n_image, n_word)
        # set_trace()
        # print(attn.values())
        #计算一个文本和128个图片之间的相似度得分
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)  #128*n
        # set_trace()
        if agg_func == 'LogSumExp':
            # row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = ops.Mul()(row_sim,lambda_lse)
            row_sim = ops.Exp()(row_sim)
            row_sim = row_sim.sum(axis=1, keepdims=True)
            # row_sim = torch.log(row_sim) / opt.lambda_lse
            row_sim = ops.Log()(row_sim) / lambda_lse   #128*1

        # elif opt.agg_func == 'Max':
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0]
        # elif opt.agg_func == 'Sum':
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        elif agg_func == 'Mean':
            # row_sim = row_sim.mean(dim=1, keepdim=True)
            row_sim = ops.ReduceMean(keep_dims=True)(row_sim,axis=1)
        # else:
        #     raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    # similarities = torch.cat(similarities, 1)
    # set_trace()
    similarities = ops.Concat(axis = 1)(similarities) # (128, 128)
    # set_trace()
    return similarities



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print,val_dataset_len=None):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()
    squeeze = ops.Squeeze()

    # switch to evaluate mode
    model.set_train(False)

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None
    max_n_word = 0
    for i, (images, captions, lengths, ids, caption_mask) in enumerate(data_loader):
        # set_trace()
        max_n_word = max(max_n_word, int(max(squeeze(lengths).asnumpy().tolist())))
#     set_trace()
    for i, (images, captions, lengths, ids, caption_mask) in enumerate(data_loader):
        #images    (128, 36, 2048)
        #captions  (128, 37)
        #lengths   (128,1)
        #ids      (128,1)
#         set_trace()
        ids = squeeze(ids).asnumpy().tolist()
        lengths = squeeze(lengths)
        # make sure val logger is used
        # model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb = model(images, captions, lengths, caption_mask)  #,val_loss
        # set_trace()
        #print(img_emb)
        if img_embs is None:
            # if img_emb.dim() == 3:
            if img_emb.dim() == 3:
                # set_trace()
                img_embs = np.zeros((val_dataset_len, img_emb.shape[1], img_emb.shape[2]))
            else:
                img_embs = np.zeros((val_dataset_len, img_emb.shape[1]))
            cap_embs = np.zeros((val_dataset_len, max_n_word, cap_emb.shape[2]))
            caption_masks = np.zeros((val_dataset_len, max_n_word))   #5000  26  
            print("--------------------------")
            cap_lens = [0] * val_dataset_len
        # cache embeddings
#         set_trace()
        #img_emb (128, 36, 1024)
        img_embs[ids] = copy.deepcopy(img_emb.asnumpy())   #(200, 36, 1024)
        lengths = lengths.asnumpy().tolist()
        max_lengths = max(lengths)
        cap_embs[ids,:max_lengths,:] = copy.deepcopy(cap_emb.asnumpy())[:, :max_lengths, :]
        caption_masks[ids,:max_lengths] = copy.deepcopy(caption_mask.asnumpy())[:, :max_lengths]
        for j, nid in enumerate(ids):
            cap_lens[nid] = int(lengths[j])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
#         set_trace()
        del images, captions
    
    return img_embs, cap_embs, caption_masks, cap_lens
#img_embs   (500, 36, 1024)
#cap_embs   (500, 58, 1024)
#cap_lens    500



def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    #加载参数
    f = open(model_path + '/config.json', 'r')  # 类型<class '_io.TextIOWrapper'>
    opt_ = json.load(f)  # 将读入的json文件转化成字典形式
    print(opt_)
    parser = argparse.ArgumentParser()

    for key,value in opt_.items():
        name = '--' + key
        parser.add_argument(name, default=value)
    opt = parser.parse_args()
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    print('Loading dataset')
    data_loader,dataset_len = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)
    # set_trace()
    # construct model
#     model = SCAN(opt,dataset_len)

    # 定义模型
    criterion = ContrastiveLoss(lambda_softmax = opt.lambda_softmax, 
                                     agg_func = opt.agg_func, 
                                     lambda_lse = opt.lambda_lse, 
                                     cross_attn = opt.cross_attn,
                                     raw_feature_norm = opt.raw_feature_norm,
                                     margin = opt.margin,
                                     max_violation = opt.max_violation)
    txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                               opt.embed_size, opt.num_layers,
                               use_bi_gru=opt.bi_gru,
                               no_txtnorm=opt.no_txtnorm,
                              batch_size=opt.batch_size)
    img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                            no_imgnorm=opt.no_imgnorm)
    testnet = BuildValNetwork(img_enc,txt_enc, criterion)
    
    #加载模型权重
    model_path = "temp/"
    image_weight_path = model_path + "image" + 'model_best.ckpt'
    text_weight_path = model_path + "text" + 'model_best.ckpt'
#     train_net.load_state_dict(image_weight_path, text_weight_path)
    image_param_dict = load_checkpoint(image_weight_path)
    load_param_into_net(img_enc, image_param_dict)
    text_param_dict = load_checkpoint(text_weight_path)
    load_param_into_net(txt_enc, text_param_dict)
    
    
    print('Computing results...')
    img_embs, cap_embs, caption_masks, cap_lens = encode_data(testnet, data_loader,val_dataset_len = dataset_len)
    # img_embs   (500, 36, 1024)
    # cap_embs   (500, 58, 1024)
    # cap_lens    500
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))
    # set_trace()

    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        start = time.time()
        if opt.cross_attn == 't2i':
            # set_trace()
            sims = shard_xattn_t2i(img_embs, 
                                   cap_embs, 
                                   caption_masks, 
                                   opt, 
                                   shard_size=100,
                                   caplens=cap_lens)
            # sims_max = sims.max(axis=1)
            #     ms.numpy.diag(sims)
            # set_trace()
        # elif opt.cross_attn == 'i2t':
        #     sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        # else:
        #     raise NotImplementedError
        end = time.time()
        print("calculate similarity time:", end-start)
        # set_trace()
        r, rt = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)


def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    return p



def shard_xattn_t2i(images, captions, caption_masks, opt, shard_size=100, caplens=None):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images)-1)/shard_size + 1)
    n_cap_shard = int((len(captions)-1)/shard_size + 1)
    #8  40
    
    d = np.zeros((len(images), len(captions)))    #(1000, 5000)
#     begin_time = time.time()
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        #0  128
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            #0  128
            # im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            # s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            im = ms.Tensor(images[im_start:im_end],ms.float32) # (128, 36, 1024)
            s = ms.Tensor(captions[cap_start:cap_end],ms.float32)  #(128, 73, 1024)
            l = ms.Tensor(caption_masks[cap_start:cap_end,:],ms.int32)
#             l = caplens[cap_start:cap_end]
            # set_trace()
            # 相似度矩阵的第一行是第一个图片对所有的文本的相似度得分
            be_time = time.time()
#             set_trace()
            sim = xattn_score_t2i_xin(im, s, l,
                                    lambda_softmax = opt.lambda_softmax,
                                    agg_func = opt.agg_func,
                                    lambda_lse = opt.lambda_lse,
                                    raw_feature_norm = opt.raw_feature_norm
                                    )
#             end_time = time.time()
#             run_time = end_time-be_time
#             print ('1该循环程序运行时间：',run_time)
            d[im_start:im_end, cap_start:cap_end] = sim.asnumpy()
#     end_time1 = time.time()
#     run_time1 = end_time1-begin_time
#     print("===================++++++++++++++++++++++")
#     print ('该循环程序运行时间：',run_time1)
    sys.stdout.write('\n')
    # set_trace()
    return d


# def shard_xattn_i2t(images, captions, caplens, opt, shard_size=128):
#     """
#     Computer pairwise i2t image-caption distance with locality sharding
#     """
#     n_im_shard = int((len(images)-1)/shard_size + 1)
#     n_cap_shard = int((len(captions)-1)/shard_size + 1)
#
#     d = np.zeros((len(images), len(captions)))
#     for i in range(n_im_shard):
#         im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
#         for j in range(n_cap_shard):
#             sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
#             cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
#             im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
#             s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
#             l = caplens[cap_start:cap_end]
#             sim = xattn_score_i2t(im, s, l, opt)
#             d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
#     sys.stdout.write('\n')
#     return d


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap   第一行表示，第一个图像对所有的文本的相似度得分
    """
    # set_trace()
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # set_trace()
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

    
"""
验证
"""
def val(model_path, data_path=None, split='dev', range_=None):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    #加载参数
    f = open(model_path + '/config.json', 'r')  # 类型<class '_io.TextIOWrapper'>
    opt_ = json.load(f)  # 将读入的json文件转化成字典形式
    print(opt_)
    parser = argparse.ArgumentParser()

    for key,value in opt_.items():
        name = '--' + key
        parser.add_argument(name, default=value)
    opt = parser.parse_args()
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    print('Loading dataset')
    data_loader,dataset_len = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)
    # set_trace()
    # construct model
#     model = SCAN(opt,dataset_len)

    # 定义模型
    criterion = ContrastiveLoss(lambda_softmax = opt.lambda_softmax, 
                                     agg_func = opt.agg_func, 
                                     lambda_lse = opt.lambda_lse, 
                                     cross_attn = opt.cross_attn,
                                     raw_feature_norm = opt.raw_feature_norm,
                                     margin = opt.margin,
                                     max_violation = opt.max_violation)
    txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                               opt.embed_size, opt.num_layers,
                               use_bi_gru=opt.bi_gru,
                               no_txtnorm=opt.no_txtnorm,
                              batch_size=opt.batch_size)
    img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                            no_imgnorm=opt.no_imgnorm)
    testnet = BuildValNetwork(img_enc,txt_enc, criterion)
    
    rsum_best = 0
    epoch_best = 0
    for i in range_:
        print("---------------------"+"  epoch： "+str(i)+"  ---------------------")
        #加载模型权重
        filename = 'checkpoint_{}.ckpt'.format(i)
        image_weight_path = model_path + "image" + filename
        text_weight_path = model_path + "text" + filename
    #     train_net.load_state_dict(image_weight_path, text_weight_path)
        image_param_dict = load_checkpoint(image_weight_path)
        load_param_into_net(img_enc, image_param_dict)
        text_param_dict = load_checkpoint(text_weight_path)
        load_param_into_net(txt_enc, text_param_dict)


        print('Computing results...')
        img_embs, cap_embs, caption_masks, cap_lens = encode_data(testnet, data_loader,val_dataset_len = dataset_len)
        # img_embs   (500, 36, 1024)
        # cap_embs   (500, 58, 1024)
        # cap_lens    500
        print('Images: %d, Captions: %d' %
              (img_embs.shape[0] / 5, cap_embs.shape[0]))
        # set_trace()

        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        start = time.time()
        if opt.cross_attn == 't2i':
            # set_trace()
            sims = shard_xattn_t2i(img_embs, 
                                   cap_embs, 
                                   caption_masks, 
                                   opt, 
                                   shard_size=100,
                                   caplens=cap_lens)
            # sims_max = sims.max(axis=1)
            #     ms.numpy.diag(sims)
            # set_trace()
        # elif opt.cross_attn == 'i2t':
        #     sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        # else:
        #     raise NotImplementedError
        end = time.time()
        print("calculate similarity time:", end-start)
        # set_trace()
        r, rt = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("epoch： "+str(i))
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
        if rsum_best < rsum:
            rsum_best = rsum
            epoch_best = i
    print("第 " + str(epoch_best) + " 代，   最高指标： "+str(rsum_best))