# from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets_recursive import PartDataset
from pointnet import RecursivePointNetCls
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--chunk_size', type=int, default=0, help='input batch size')
parser.add_argument('--overlap', action='store_true', default=False,
                    help='Overlapping chunks.')

opt = parser.parse_args()
print (opt)

print '**************** overlap: ', opt.overlap

if opt.overlap:
	overlap_size = opt.chunk_size/2
else:
	overlap_size = 0

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', npoints_per_chunk = opt.chunk_size, train = False, classification = True, overlap_size=overlap_size)

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle = False)


classifier = RecursivePointNetCls(k = len(test_dataset.classes), num_points_per_chunk = opt.chunk_size)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

total_correct=0
total_count  =0
for i, data in enumerate(testdataloader, 0):
	points, target = data
	target_input = target.numpy()
	points_input = points.view(-1,2500, 3)#torch.squeeze(points, dim=1)
	points_input = points_input.cpu().numpy()
	points, target = Variable(points), Variable(target[:,0])
	points = points.transpose(3,2)
	points, target = points.cuda(), target.cuda()
	pred, _ = classifier(points)
	loss = F.nll_loss(pred, target)
	pred_choice = pred.data.max(1)[1]
	correct = pred_choice.eq(target.data).cpu().sum()
	print('i:%d  loss: %f accuracy: %f' %(i, loss.data[0], correct/float(points_input.shape[0])))
	total_correct+=correct
	total_count+=float(points_input.shape[0])

	print('prediction: ', pred_choice.cpu().numpy()[0] ,', target: ', target_input[0])


print 'overall accuracy: ', total_correct/total_count
