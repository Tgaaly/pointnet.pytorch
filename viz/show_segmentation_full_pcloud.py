# _future__ import print_function
import argparse
import os, sys
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
from datasets import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from data_kitti import FileListDataset
from data_kitti import HDF5Dataset
from data_kitti import DataSplitter

import pdb
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')


opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.batchSize=opt.batch_size

hdf5_file = '/home/tarek/Data/training.hdf5'
data_splitter = DataSplitter(hdf5_file)
# Setup training and validation data loaders.
flag_data_augmentation = False#config.flag_data_augmentation#False
dataset = HDF5Dataset(hdf5_file, data_splitter, 'testing', flag_data_augmentation=flag_data_augmentation)
# hdf5_dataset_val = HDF5Dataset(hdf5_file, data_splitter, 'validation', flag_data_augmentation=False, flag_normalize=flag_normalize)

# Weighted sampling: sample pedestrians more than vehicles.
# sampler = sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)#, num_workers=2)


# dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                           shuffle=True, num_workers=int(opt.workers))

# # test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'], train = False)
# testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
#                                           shuffle=True, num_workers=int(opt.workers))

print(len(dataset))#, len(test_dataset))
num_classes = 3#dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x:'\033[94m' + x + '\033[0m'


classifier = PointNetDenseCls(k = num_classes, num_points=2500)
classifier.cuda()
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
else:
    print 'must provide model'
    sys.exit()

optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

classifier.eval()

num_batch = len(dataset)/opt.batchSize
fig=plt.figure(1)

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        
        points, target = Variable(points), Variable(target)
        points = points.transpose(2,1) 

        points, target = points.cuda(), target.cuda()   
        optimizer.zero_grad()
        pred, _ = classifier(points)

        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0]# - 1

        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] test loss: %f accuracy: %f' %(epoch, i, num_batch, loss.data[0], correct/float(opt.batchSize * 2500)))
        

        points_numpy = points.data.cpu().numpy()
        pred_choice = pred_choice.cpu().numpy()
        
        indices = np.where(pred_choice==0)
        class1_pts = points_numpy[0,:,indices[0]]
        indices = np.where(pred_choice==1)
        class2_pts = points_numpy[0,:,indices[0]]
        indices = np.where(pred_choice==2)
        class3_pts = points_numpy[0,:,indices[0]]
        # pdb.set_trace()
        print '#background', class1_pts.shape
        print '#pedestrians', class2_pts.shape
        print '#vehicles', class3_pts.shape

        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        ax.scatter(class1_pts[:,0], class1_pts[:,1], class1_pts[:,2], color='red')
        ax.scatter(class2_pts[:,0], class2_pts[:,1], class2_pts[:,2], color='green')
        ax.scatter(class3_pts[:,0], class3_pts[:,1], class3_pts[:,2], color='blue')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.pause(5)

