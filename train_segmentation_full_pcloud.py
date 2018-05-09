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
from data.datasets import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_kitti import FileListDataset
from data.data_kitti import HDF5Dataset
from data.data_kitti import DataSplitter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.training_utils import print_colors

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--outdir', type=str, default='seg_full_pcloud_scaled01_largernet',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--lr', default = 0.0008)

opt = parser.parse_args()
print (opt)

opt.outdir = os.path.join('results', opt.outdir)

def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	for param_group in optimizer.param_groups:
		old_lr = param_group['lr']

	new_lr = opt.lr * (0.1 ** (epoch // 10))

	if np.abs(new_lr-old_lr) > 1e-7:
		print '>>> Decreasing LR from: {} to {}'.format(old_lr, new_lr)
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr

# def one_hot(index, classes):
#     size = index.size() + (classes,)
#     view = index.size() + (1,)
#     mask = torch.Tensor(*size).fill_(0)
#     index = index.view(*view)
#     ones = 1.
#     if isinstance(index, Variable):
#         ones = Variable(torch.Tensor(index.size()).fill_(1))
#         mask = Variable(mask, volatile=index.volatile)
#     return mask.scatter_(1, index, ones)

# def focal_loss(x, y):
#     # pdb.set_trace()
#     # print x.size()
#     y = one_hot(y.cpu(), x.size(-1)).cuda()
#     #logit = x #F.softmax(x)
#     x = x.clamp(1e-7, 1. - 1e-7)
#     # cross_entropy
#     # cross_entropy = -.99 * 
#     cross_entropy = -1.0 * y.float() * torch.log(x) 
#     cross_entropy = cross_entropy * (1.0 - x) ** 0.01
#     print cross_entropy.size()
#     loss = torch.sum(cross_entropy, 1)# * (1.0 - x) ** 2.0
#     print loss.size()
#     #pdb.set_trace()
#     loss = loss.sum() / (loss.size(0))
    
#     return loss

# def focal_loss(output, target):
# 	alpha = 0.75
# 	gamma = 2.0
# 	print output.size()
# 	print target.size()
# 	pdb.set_trace()
# 	focal_loss_1 = -alpha * torch.pow((1 - output[:,0]), gamma) * torch.log(output[:,0])
# 	focal_loss_2 = -alpha * torch.pow((1 - output[:,1]), gamma) * torch.log(output[:,1])
	# return 0.0

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.batchSize=opt.batch_size

hdf5_file = '/home/tarek/Data/training.hdf5'
data_splitter = DataSplitter(hdf5_file)
# Setup training and validation data loaders.
flag_data_augmentation = False
dataset = HDF5Dataset(hdf5_file, data_splitter, 'training', flag_data_augmentation=flag_data_augmentation)

dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

NUM_POINTS = 40000
print(len(dataset))
num_classes = 2#3#dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outdir)
except OSError:
    pass

classifier = PointNetDenseCls(k = num_classes, num_points=NUM_POINTS)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.SGD(classifier.parameters(), lr=opt.lr, momentum=0.9)
classifier.cuda()
fig=plt.figure(1)
num_batch = len(dataset)/opt.batchSize
print 'num_batch = ', num_batch

loss_train=[]

for epoch in range(opt.nepoch):

    # adjust_learning_rate(optimizer, epoch)

    for batch_idx, data in enumerate(dataloader, 0):
        points, target_orig = data

        points, target_orig = Variable(points), Variable(target_orig)
        points = points.transpose(2,1) 

        points, target_orig = points.cuda(), target_orig.cuda()   
        optimizer.zero_grad()
        pred_, trans64 = classifier(points)

        pred = pred_.view(-1, num_classes)
        target = target_orig.view(-1,1)[:,0]# - 1
        
        # (Weighted by class) negative log likelihood loss.
        class_weights = torch.zeros(2)
        class_weights[1] = 0.975#0.92#0.94
        class_weights[0] = 1.0-class_weights[1]

        current_batch_size = target_orig.size(0)
        batch_iden = Variable(torch.from_numpy(np.array(np.eye(64)).astype(np.float32))).view(1,64,64).repeat(current_batch_size,1,1).cuda()
        AA_transpose = torch.bmm(trans64, trans64.transpose(1,2))

        # Frobenius norm (elementwise L2 norm --> 
        # square root of the sum of squared element-wise differences between two matrices.
        diff = torch.abs(batch_iden - AA_transpose)
        diff_sq = diff.pow(2)
        diff_sum = diff_sq.sum()
        
        loss_reg = 0.001*diff_sum

        loss = F.nll_loss(pred, target, weight=class_weights.cuda()) + loss_reg
        # loss = focal_loss(pred, target)

        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        # pdb.set_trace()
        print('[%d: %d/%d] train loss: %s accuracy: %f' %(epoch, batch_idx, num_batch, print_colors.blue(str(loss.data[0])), correct/float(pred_choice.size(0))))
        
        loss_train.append(loss.data[0])
        np.savez(os.path.join(opt.outdir,'train_loss.npz'), loss_train=loss_train)

        if batch_idx % 10 == 0:
	        points_numpy = points.data[0,:,:].cpu().numpy()
	        pred_choice = pred_.data[0,:,:].cpu().numpy().argmax(1)#pred_.data[0,:,:].max(1)[1].cpu().numpy()
	        indices = np.where(pred_choice==0)
	        class1_pts = points_numpy[:,indices[0]]
	        indices = np.where(pred_choice==1)
	        class2_pts = points_numpy[:,indices[0]]
	        indices = np.where(pred_choice==2)
	        class3_pts = points_numpy[:,indices[0]]

	        plt.clf()
	        ax = fig.add_subplot(121)#, projection='3d')
	        ax.set_aspect('equal')
	        ax.scatter(class1_pts[0,:], class1_pts[1,:], color='red', s=1)
	        ax.scatter(class2_pts[0,:], class2_pts[1,:], color='green', s=1)
	        ax.scatter(class3_pts[0,:], class3_pts[1,:], color='blue', s=1)
	        # ax.scatter(class1_pts[0,:], class1_pts[1,:], class1_pts[2,:], color='red')
	        # ax.scatter(class2_pts[0,:], class2_pts[1,:], class2_pts[2,:], color='green')
	        # ax.scatter(class3_pts[0,:], class3_pts[1,:], class3_pts[2,:], color='blue')
	        ax.set_title('prediction')
	        ax.set_xlabel('X Label')
	        ax.set_ylabel('Y Label')
	        # ax.set_zlabel('Z Label')

	        targets_numpy = target_orig.data[0,:,:].cpu().numpy()
	        indices = np.where(targets_numpy==0)
	        class1_pts = points_numpy[:,indices[0]]
	        indices = np.where(targets_numpy==1)
	        class2_pts = points_numpy[:,indices[0]]
	        indices = np.where(targets_numpy==2)
	        class3_pts = points_numpy[:,indices[0]]
	        ax = fig.add_subplot(122)#, projection='3d')
	        ax.set_aspect('equal')
	        ax.scatter(class1_pts[0,:], class1_pts[1,:], color='red', s=1)
	        ax.scatter(class2_pts[0,:], class2_pts[1,:], color='green', s=1)
	        ax.scatter(class3_pts[0,:], class3_pts[1,:], color='blue', s=1)
	        ax.set_title('label')
	        ax.set_xlabel('X Label')
	        ax.set_ylabel('Y Label')

	        plt.savefig(os.path.join(opt.outdir,'results_')+str(epoch)+'_'+str(batch_idx)+'.png')

    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outdir, epoch))