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
from data.datasets import Modelnet40
from pointnet import PointNetCls
import torch.nn.functional as F

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
# parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

# opt.manualSeed = random.randint(1, 10000) # fix seed
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)

# Path to modelnet40 dataset. Download from http://modelnet.cs.princeton.edu/
data_dir = '/home/tarek/Data/ModelNet40/ModelNet40'

# Test set data loader.
test_dataset = Modelnet40(root = data_dir, classification = True, train = False, npoints = opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

# print(len(dataset), len(test_dataset))
print(len(test_dataset))
num_classes = len(test_dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# Declare pointnet model.
classifier = PointNetCls(k = num_classes, num_points = opt.num_points)

# Load saved checkpoint.
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

# Copy and setup to run model on GPU.
classifier.cuda()
# Set model for eval mode.
classifier.eval()

# Stats.
correct = 0
num = 0
best_test_acc=0.0

# Loop through test set.
for i, data in enumerate(testdataloader, 0):
    points, target = data
    points, target = Variable(points, volatile=True), Variable(target[:,0])
    points = points.transpose(2,1)
    points, target = points.cuda(), target.cuda()
    pred, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct += pred_choice.eq(target.data).cpu().sum()
    num += int(target.data.size(0))
    print 'batch ', i

# if (correct/float(num)) > best_test_acc:
best_test_acc = correct/float(num)
# print 'updating best test accuracy...'

print '# of testing = ', num
print 'best_test_acc: ', best_test_acc