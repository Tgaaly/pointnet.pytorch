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

import matplotlib.cm as cm

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')


opt = parser.parse_args()
print (opt)

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0' , train = False, classification = True)

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle = False)


# classifier = RecursivePointNetCls(k = len(test_dataset.classes))
# classifier.cuda()
# classifier.load_state_dict(torch.load(opt.model))
# classifier.eval()

total_correct=0
total_count  =0
for i, data in enumerate(testdataloader, 0):
	points, target = data
	# points_input = points.cpu().numpy()
	target_input = target.numpy()
	# print 'target shape: ', target_input.shape
	# print '>', points.size()
	# points_input = points.view(-1,2500, 3)#torch.squeeze(points, dim=1)
	points_input = points
	points_input = points_input.cpu().numpy()
	# print points_input.shape
	# # points_input = points_input[:,::5,:]
	# print points_input.shape
	# # points = torch.FloatTensor(points_input)
	# points, target = Variable(points), Variable(target[:,0])
	# #points = points.transpose(2,1)
	# points = points.transpose(3,2)
	# points, target = points.cuda(), target.cuda()
	# pred, _ = classifier(points)
	# loss = F.nll_loss(pred, target)
	# pred_choice = pred.data.max(1)[1]
	# correct = pred_choice.eq(target.data).cpu().sum()
	# print('i:%d  loss: %f accuracy: %f' %(i, loss.data[0], correct/float(32)))
	# total_correct+=correct
	# total_count+=float(32)
	print i
	# print('prediction: ', pred_choice.cpu().numpy()[0] ,', target: ', target_input[0])

	print test_dataset.classes
	# print '>>>', points_input.shape
	# # print(points_input.shape)
	# print target_input[0][0]
	# print points_input.shape
	# print target_input
	indices = np.where(target_input==3)[0]
	# print indices
	if len(indices)>0:#2 in target_input:# ==2:
		
		# colors = iter(cm.rainbow(np.linspace(0, 1, 25)))
		for j in indices:
			# j = indices[0]
			print j
			fig=plt.figure(1)
			plt.clf()
			ax = fig.add_subplot(111, projection='3d')
			ax.set_aspect('equal')

			colors = cm.Dark2(np.linspace(0, 1, 25))
			# colors = colors[np.random.randint(len(colors),...)]
			# print colors
			colors[:,3]=1.0
			for p in xrange(25):
				ax.scatter(points_input[j,p,:,0], points_input[j,p,:,1], points_input[j,p,:,2], color=colors[p,:], alpha=1.0)#next(colors))
			# plt.title('prediction: '+str(pred_choice.cpu().numpy()[0][0])+', target: '+str(target_input[0][0]))
			ax.set_xlabel('X Label')
			ax.set_ylabel('Y Label')
			ax.set_zlabel('Z Label')
			plt.pause(5)
