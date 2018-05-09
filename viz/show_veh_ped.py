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
from pointnet import PointNetCls_veh_ped
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import FileListDataset
from data import HDF5Dataset
from data import DataSplitter
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')


opt = parser.parse_args()
print (opt)

# test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0' , train = False, classification = True)
# testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle = False)

batch_size=1
# Data splitting
hdf5_file = '/home/tarek/Data/training.hdf5'
data_splitter = DataSplitter(hdf5_file)
hdf5_dataset = HDF5Dataset(data_splitter, 'testing', flag_data_augmentation=False, flag_with_intensities=False)
test_loader = DataLoader(hdf5_dataset, batch_size=batch_size, shuffle=False)


classifier = PointNetCls_veh_ped(k = 2)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

count_ped=0
count_veh=0
total_correct = 0
total_count = 0

for i, data in enumerate(test_loader):
	points = data['points']
	target = data['targets']
	points_input = points.cpu().numpy()
	target_input = target.cpu().numpy()

	# print points_input.shape
	# points_input = points_input[:,::5,:]
	points = torch.FloatTensor(points_input)
	points, target = Variable(points), Variable(target[:,0])
	points = points.transpose(2,1)
	points, target = points.cuda(), target.cuda()
	pred, _ = classifier(points)
	loss = F.nll_loss(pred, target)
	pred_choice = pred.data.max(1)[1]
	correct = pred_choice.eq(target.data).cpu().sum()
	total_correct+=correct
	total_count+=batch_size
	# print('i:%d  loss: %f accuracy: %f' %(i, loss.data[0], correct/float(batch_size)))

	the_prediction = np.asscalar(pred_choice.cpu().numpy())
	# if np.asscalar(target_input)==1:
	# count_ped+=1
	# print('prediction: ', the_prediction, 'target: ', target_input)
	# print pred_choice.cpu().numpy()
	# # print(np.asscalar(pred_choice.cpu().numpy()[0]))
	if points_input.shape[1]>300:
		fig=plt.figure(1)
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(points_input[0,:,0], points_input[0,:,1], points_input[0,:,2])
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		plt.title('prediction: ' +  str(the_prediction) + ' - target: ' +  str(np.asscalar(target_input)))
		plt.pause(5)
	# else:
		# count_veh+=1
	
print('#vehicles: ', count_veh, ' pedestrians: ', count_ped)
print 'overall accuracy: ', total_correct / float(total_count)