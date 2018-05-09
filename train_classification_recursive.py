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
from data.datasets_recursive import PartDataset
from pointnet import RecursivePointNetCls
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--chunk_size', type=int, default=0, help='chunk size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--overlap', action='store_true', default=False, help='Overlapping chunks.')

opt = parser.parse_args()
print (opt)

opt.outf = []

if opt.overlap:
    overlap_sz = opt.chunk_size/2
    opt.outf = 'rec-blstm-'+str(opt.chunk_size)+'_overlap'
else:
    overlap_sz = 0
    opt.outf = 'rec-blstm-'+str(opt.chunk_size)

print '**********************************************************************'
print 'chunk_size = ', opt.chunk_size
print 'out_dir = ', opt.outf
print 'overlap = ', opt.overlap
print 'overlap_sz = ', overlap_sz
print '**********************************************************************'

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, npoints_per_chunk = opt.chunk_size, overlap_size=overlap_sz)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, train = False, npoints_per_chunk = opt.chunk_size, overlap_size=overlap_sz)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


classifier = RecursivePointNetCls(k = num_classes, num_points_per_chunk = opt.chunk_size)


if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
classifier.cuda()

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data

        points, target = Variable(points), Variable(target[:,0])
        points = points.transpose(3,2)

        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()

        pred, _ = classifier(points)

        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.data[0], correct/float(opt.batchSize)))

        if i % 10 == 0:
            j, data = enumerate(testdataloader, 0).next()
            points, target = data
            points, target = Variable(points), Variable(target[:,0])
            points = points.transpose(3, 2)
            points, target = points.cuda(), target.cuda()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('>>>>>>>>>>> [%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.data[0], correct/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
