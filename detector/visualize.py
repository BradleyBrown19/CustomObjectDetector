import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from efficientdet.losses import FocalLoss

from efficientdet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer

from fastai.basic_train import Learner
from fastai.basic_data import DataBunch, Dataset

from matplotlib import pyplot as plt

from efficientdet.efficientdet import EfficientDet

from utils import EFFICIENTDET, get_state_dict


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--model', help='Path to model (.pt) file.')

	parser.add_argument('--model_type', help="Either fastai or pytorch", default="fastai")

	parser.add_argument('--threshold', help="Threshold for prediction", type=float, default=0.5)

	parser.add_argument('--network', default='efficientdet-d0', type=str,
                    help='efficientdet-[d0, d1, ..]')
	
	parser.add_argument('--scales', nargs='+', default=[8, 16, 32], type=float,
                    help='Scales for anchor box config')
					
	parser.add_argument('--ratios', nargs='+', default=[0.5, 1.0, 2.0], type=float,
                    help='Ratios for anchor box config')

	parser = parser.parse_args(args)

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'csv':
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
	
	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	efficientdet = EfficientDet(num_classes=dataset_val.num_classes(),
                         network=parser.network,
                         scales=parser.scales,
                         ratios=parser.ratios,
                         W_bifpn=EFFICIENTDET[parser.network]['W_bifpn'],
                         D_bifpn=EFFICIENTDET[parser.network]['D_bifpn'],
                         D_class=EFFICIENTDET[parser.network]['D_class']
                         )

	state_dict = torch.load(parser.model, map_location=torch.device('cpu'))

	if parser.model_type == "fastai":
		efficientdet.load_state_dict(state_dict['model'])
	elif parser.model_type == "pytorch":
		efficientdet.load_state_dict(state_dict)
	else:
		print("UNSUPPORTED MODEL TYPE")

	efficientdet.inference = True

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	for idx, data in enumerate(dataloader_val):
		data = {'img': data[0], 'annot': data[1]}

		with torch.no_grad():
			st = time.time()
			if torch.cuda.is_available():
				scores, classification, transformed_anchors = efficientdet((data['img'].cuda().float(), data['annot'].cuda()))
			else:
				scores, classification, transformed_anchors = efficientdet((data['img'].float(), data['annot']))
			#print('Elapsed time: {}'.format(time.time()-st))
			idxs = np.where(scores.cpu()>parser.threshold)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				print(label_name)

			plt.figure(figsize=(20,10))
			plt.imshow(img)
			plt.show()



if __name__ == '__main__':
 main()