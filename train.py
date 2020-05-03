import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from efficientdet import efficientdet
from efficientdet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from efficientdet import coco_eval
from efficientdet import csv_eval

from tqdm import tqdm
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import EFFICIENTDET, get_state_dict
from efficientdet import losses
from efficientdet import callbacks

import fastai
from fastai.basic_data import DataBunch, Dataset
from fastai import *
from fastai.vision import *
from fastai.basic_train import Learner

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--bs', help="batch size", type=int, default=8)

    parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--save_on_improvement', default=True, type=str,
                    help='Save model every decrease in validation loss')
    parser.add_argument('--wd', default=1e-4,
                    help='Weight decay')
    parser.add_argument('--num_epochs', default=1,
                    help='Weight decay')
    parser.add_argument('--lr', default=1e-5,
                    help='Weight decay')
    parser.add_argument('--weighted_loss', default=None, help="Weighted loss for classification")

    parser.add_argument('--network', default='efficientdet-d0', type=str,
                    help='efficientdet-[d0, d1, ..]')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
        sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.bs, drop_last=False)
        dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
        data = DataBunch(dataloader_train, dataloader_val, collate_fn=collater)

    elif parser.dataset == 'csv':

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Resizer()]))
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
        sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.bs, drop_last=False)
        dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
        data = DataBunch(dataloader_train, dataloader_val, collate_fn=collater)

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    model = EfficientDet(num_classes=dataset_train.num_classes(),
                         network=args.network,
                         W_bifpn=EFFICIENTDET[args.network]['W_bifpn'],
                         D_bifpn=EFFICIENTDET[args.network]['D_bifpn'],
                         D_class=EFFICIENTDET[args.network]['D_class']
                         )
    
    if(args.resume is not None):
        model.load_state_dict(checkpoint['state_dict'])

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model.inference = False

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    loss = losses.FocalLoss(num_classes=dataset_train.num_classes(), weights=parser.weighted_loss, gpu=torch.cuda.is_available())

    learn = Learner(data, retinanet, loss_func=loss)

    callbacks = []

    if args.save_on_improvement:
        callbacks.append(CustomSaveModelCallback(learn))

    learn.fit(parser.num_epochs, parser.lr, parser.weight_decay, callbacks=callbacks)

    learn.save("final_model")


if __name__ == '__main__':
    main()
