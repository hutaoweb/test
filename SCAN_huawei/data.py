# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""
# import torch
# import torch
# import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
from ipdb import set_trace
import mindspore.dataset as ds


# import torch.utils.data as data

class PrecompDataset:
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
#         self.captions = []
#         with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
#             for line in f:
#                 self.captions.append(line.strip())
        caption_path = "./" + data_split + "_captions_words.npy"
        # caption_path = data_split + "_captions_words.npy"
        self.captions = np.load(caption_path,allow_pickle=True)


        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for cocos is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000


    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        image = self.images[img_id]
#         caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        # set_trace()
#         tokens = nltk.tokenize.word_tokenize(str(caption).lower().encode().decode('utf-8'))
        tokens = self.captions[index]
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = caption
        caption_mask = np.zeros((87))
        caption_mask[:len(caption)] += 1 
#         caption_mask[:87,:] += 1

        image = np.array(image).astype(np.float32)
        target = np.array(target).astype(np.int32)
        ids = np.array([index]).astype(np.int32)
        lengths = np.array([len(caption)]).astype(np.int32)
        caption_mask = caption_mask.astype(np.int32)
        return image,  target, lengths, ids, caption_mask

    def __len__(self):
        return self.length
#



def unbatch_concat_padded(captions):
#     idx = np.argwhere(np.all(captions[..., :] == 0, axis=0))
#     captions = np.delete(captions, idx, axis=1)
    captions = captions[:,:87]
    return captions

# def unbatch_concat_padded(images, captions, lengths, ids ):
#     idx = np.argwhere(np.all(captions[..., :] == 0, axis=0))
#     captions = np.delete(captions, idx, axis=1)
#
#     order = np.array([15, 30, 35, 60, 31, 25, 36, 85, 26, 37, 27, 55, 110, 45, 80, 10, 51, 86, 125, 5, 28, 61, 120, 126, 6, 12, 16, 38, 50, 75, 90, 17, 20, 32, 56, 111, 115, 39, 47, 48, 76, 100, 105, 112, 116, 118, 121, 123, 127, 11, 13, 40, 53, 82, 87, 95, 1, 2, 8, 21, 42, 52, 57, 70, 81, 107, 113, 117, 122, 0, 7, 22, 33, 41, 63, 84, 91, 93, 101, 102, 3, 18, 29, 46, 54, 62, 65, 66, 79, 83, 89, 4, 9, 49, 58, 67, 77, 78, 97, 106, 108, 119, 14, 92, 94, 124, 23, 44, 69, 71, 72, 96, 98, 24, 64, 68, 88, 109, 114, 34, 43, 59, 73, 74, 103, 104, 19, 99])
#
#     # lengths = -lengths.flatten()
#     # print("lengths",lengths)
#     # print("ids",ids)
#     # order = np.argsort(lengths)
#     # print("order", order)
#     images = images[order]
#     captions = captions[order]
#     ids = ids[order].flatten()
#     lengths = lengths[order].flatten()
#     # set_trace()
#     return images, captions, lengths, ids



def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)
    # set_trace()
    if data_split == "train":
        drop_remainder = True
    else:
        drop_remainder = False
    data_loader = ds.GeneratorDataset(dset,
                                      ["images", "captions","lengths", "ids", "caption_mask"],   #
                                      shuffle=shuffle)
    data_loader = data_loader.batch(batch_size = batch_size,
                      pad_info={"captions":([87], 0)},
                      drop_remainder = drop_remainder)
    # data_loader = data_loader.map(input_columns=["images", "captions","lengths", "ids"], operations=unbatch_concat_padded)
#     data_loader = data_loader.map(input_columns=["captions"], operations=unbatch_concat_padded)
    return data_loader,len(dset)


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader,train_dataset_len = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader, val_dataset_len = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader,train_dataset_len,val_dataset_len


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader,dataset_len = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader,dataset_len
