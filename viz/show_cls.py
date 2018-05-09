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
from datasets import PartDataset
from pointnet import PointNetCls
import torch.nn.functional as F
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')


opt = parser.parse_args()
print (opt)

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0' , train = False, classification = True)

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle = False)


classifier = PointNetCls(k = len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()
total_correct=0
total_count=0
for i, data in enumerate(testdataloader, 0):
	points, target = data
	points_input = points.cpu().numpy()
	print points_input.shape
	# points_input = points_input[:,::5,:]
	print points_input.shape
	points = torch.FloatTensor(points_input)
	points, target = Variable(points), Variable(target[:,0])
	points = points.transpose(2,1)
	points, target = points.cuda(), target.cuda()
	pred, _ = classifier(points)
	loss = F.nll_loss(pred, target)
	pred_choice = pred.data.max(1)[1]
	correct = pred_choice.eq(target.data).cpu().sum()
	print('i:%d  loss: %f accuracy: %f' %(i, loss.data[0], correct/float(32)))

	total_correct+=correct
	total_count+=float(32)
	print(points_input.shape)
	#fig=plt.figure(1)
	#ax = fig.add_subplot(111, projection='3d')
	#ax.scatter(points_input[0,:,0], points_input[0,:,1], points_input[0,:,2])
	#ax.set_xlabel('X Label')
	#ax.set_ylabel('Y Label')
	#ax.set_zlabel('Z Label')
	#plt.pause(2)

print('overall accuracy=',total_correct/total_count)
