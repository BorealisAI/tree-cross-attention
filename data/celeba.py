# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2021, Tung Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TNP (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/tung-nd/TNP-pytorch by Tung Nguyen
####################################################################################


import torch
import os.path as osp

from utils.paths import datasets_path


class CelebA(object):
    def __init__(self, train=True, resolution=32):
        self.data, self.targets = torch.load(
            osp.join(
                datasets_path, f"celeba{resolution}", "train.pt" if train else "eval.pt"
            )
        )
        self.data = self.data.float() / 255.0

        if train:
            self.data, self.targets = self.data, self.targets
        else:
            self.data, self.targets = self.data, self.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


if __name__ == "__main__":

    # preprocess
    # before proceeding, download img_align_celeba.zip from
    # https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg and unzip
    # ,download list_eval_partitions.txt from
    # https://drive.google.com/drive/folders/0B7EVK8r0v71pdjI3dmwtNm5jRkE
    # and download identity_CelebA.txt from
    # https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs
    # and place them in ${datasets_path}/celeba folder.

    import os
    import os.path as osp
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
    import torch

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=32)
    args = parser.parse_args()

    # load train/val/test split
    splitdict = {}
    with open(osp.join(datasets_path, "celeba", "list_eval_partition.txt"), "r") as f:
        for line in f:
            fn, split = line.split()
            splitdict[fn] = int(split)

    # load identities
    iddict = {}
    with open(osp.join(datasets_path, "celeba", "identity_CelebA.txt"), "r") as f:
        for line in f:
            fn, label = line.split()
            iddict[fn] = int(label)

    train_imgs = []
    train_labels = []
    eval_imgs = []
    eval_labels = []
    path = osp.join(datasets_path, "celeba", "img_align_celeba")
    imgfilenames = os.listdir(path)
    for fn in tqdm(imgfilenames):

        img = Image.open(osp.join(path, fn)).resize((args.resolution, args.resolution))
        if splitdict[fn] == 2:
            eval_imgs.append(torch.LongTensor(np.array(img).transpose(2, 0, 1)))
            eval_labels.append(iddict[fn])
        else:
            train_imgs.append(torch.LongTensor(np.array(img).transpose(2, 0, 1)))
            train_labels.append(iddict[fn])

    print(f"{len(train_imgs)} train, {len(eval_imgs)} eval")

    train_imgs = torch.stack(train_imgs)
    train_labels = torch.LongTensor(train_labels)
    torch.save(
        [train_imgs, train_labels],
        osp.join(datasets_path, f"celeba{args.resolution}", "train.pt"),
    )

    eval_imgs = torch.stack(eval_imgs)
    eval_labels = torch.LongTensor(eval_labels)
    torch.save(
        [eval_imgs, eval_labels],
        osp.join(datasets_path, f"celeba{args.resolution}", "eval.pt"),
    )
