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
# from datasets import PartDataset
from pointnet import PointNetCls_veh_ped
import torch.nn.functional as F

from data import FileListDataset
from data import HDF5Dataset
from data import DataSplitter

from torch.utils.data import sampler
from torch.utils.data import DataLoader

import config

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=config.num_points, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='veh_ped',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, npoints = opt.num_points)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                           shuffle=True, num_workers=int(opt.workers))

# test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, train = False, npoints = opt.num_points)
# testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
#                                           shuffle=True, num_workers=int(opt.workers))

# Data splitting.
hdf5_file = '/home/tarek/Data/training.hdf5'
data_splitter = DataSplitter(hdf5_file)
sample_weights = data_splitter.sample_weights
print '#pedestrians found: ', data_splitter.num_of_pedestrians
print '#vehicles found: ', data_splitter.num_of_vehicles

# Setup training and validation data loaders.
flag_data_augmentation = False
flag_with_intensities = False
hdf5_dataset = HDF5Dataset(data_splitter, 'training', flag_data_augmentation=flag_data_augmentation, flag_with_intensities=flag_with_intensities)
hdf5_dataset_val = HDF5Dataset(data_splitter, 'validation', flag_data_augmentation=False, flag_with_intensities=flag_with_intensities)

# Weighted sampling: sample pedestrians more than vehicles.
sampler = sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(hdf5_dataset, batch_size=opt.batchSize, sampler=sampler, shuffle=False)#, num_workers=2)
val_loader = DataLoader(hdf5_dataset_val, batch_size=opt.batchSize, shuffle=False)


# print(len(dataset), len(test_dataset))
num_classes = 2
# print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


classifier = PointNetCls_veh_ped(k = num_classes, num_points = config.num_points)


if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda()

num_batch = len(train_loader)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(train_loader):
        # points, target = data
        points = data['points']
        target = data['targets']

        # print points.size()
        # print target.size()
        # print type(points)
        # print type(target)

        # print target.cpu().numpy()
        points, target = Variable(points), Variable(target[:,0])
        points = points.transpose(2,1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        pred, _ = classifier(points)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        if i % 10 == 0:
            print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.data[0], correct/float(opt.batchSize)))

        if i % 100 == 0:
            j, data = enumerate(val_loader).next()
            # points, target = data
            points = data['points']
            target = data['targets']
            points, target = Variable(points), Variable(target[:,0])
            points = points.transpose(2,1)
            points, target = points.cuda(), target.cuda()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('validation'), loss.data[0], correct/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
