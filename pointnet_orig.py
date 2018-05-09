import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F

class STN3d_64(nn.Module):
    def __init__(self, num_points = None):
        super(STN3d_64, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4096)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        tmatrix_flattened = np.eye(64).flatten()
        iden = Variable(torch.from_numpy(np.array(tmatrix_flattened).astype(np.float32))).view(1,4096).repeat(batchsize,1)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 64, 64)
        return x


class STN3d(nn.Module):
    def __init__(self, num_points = None):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

# Main Classification Network.
class PointNetfeat(nn.Module):
    def __init__(self, num_points = None, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.stn64 = STN3d_64(num_points = num_points)

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)

        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        # self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        # self.bn6 = nn.BatchNorm1d(1024)

        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):

        batchsize = x.size()[0]        
        # Input transform (3x3 matrix multiplication).
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)

        # MLP(64,64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Feature transform (64x64 matrix multiplication).
        trans64 = self.stn64(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans64)
        x = x.transpose(2,1)

        # Point feature (nx64) used in segmentation net.
        # pointfeat = x
        
        x = F.relu(self.bn3(self.conv3(x)))


        x = F.relu(self.bn4(self.conv4(x)))
        pointfeat = x

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans64
        else:
            # concatenate global feature with all the pointfeatures.
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans64


class PointNetCls(nn.Module):
    def __init__(self, num_points = None, k = 2):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        # Dropout layers
        self.do1 = nn.Dropout(p=0.7)
        self.do2 = nn.Dropout(p=0.7)

    def forward(self, x):
        x, trans3, trans64 = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do2(x)

        x = self.fc3(x)
        # for focal loss
        return x, trans3, trans64#F.softmax(x), trans#
        # for CE
        #return F.log_softmax(x), trans


# Segmentation network part
class PointNetDenseCls(nn.Module):
    def __init__(self, num_points = None, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k 
        self.feat = PointNetfeat(num_points, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1024+128, 512, 1)#1088
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 64, 1)
        self.conv5 = torch.nn.Conv1d(64, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)

        x = x.transpose(2,1).contiguous()
        # x = F.softmax(x.view(-1,self.k))
        x = F.log_softmax(x.view(-1,self.k))
        x = x.view(batchsize, self.num_points, self.k)
        return x, trans



class RecursivePointNetCls(nn.Module):
    def __init__(self, num_points_per_chunk=None, k = 2):
        super(RecursivePointNetCls, self).__init__()
        self.feat = RecursivePointNetfeat(num_points_per_chunk=num_points_per_chunk, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        # recursive part.
        input_size = 1024
        hidden_size = 128#128
        self.rnn1 = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        self.fclast = nn.Linear(hidden_size*2, 16)

    def forward(self, x):
        batchsize = x.size()[0]

        x_seq = []
        for i in xrange(x.size()[1]):
            x_slice = x[:,i,:,:]
            # run on slice.
            x_slice_feat, trans = self.feat(x_slice)
            x_seq.append(x_slice_feat)

        x_seq_stacked = torch.stack(x_seq, dim=1)
        x_rnn_out, h_rnn1 = self.rnn1(x_seq_stacked)

        # Take last output of sequence as class label.
        x_output = x_rnn_out[:,-1,:]
        x_output = self.fclast(x_output)

        return F.log_softmax(x_output), trans


class RecursiveSTN3d(nn.Module):
    def __init__(self, num_points_per_chunk = None):
        super(RecursiveSTN3d, self).__init__()
        self.num_points_per_chunk = num_points_per_chunk
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points_per_chunk)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)

        return x

class RecursivePointNetfeat(nn.Module):
    def __init__(self, num_points_per_chunk=None, global_feat = True):
        super(RecursivePointNetfeat, self).__init__()
        self.stn = RecursiveSTN3d(num_points_per_chunk = num_points_per_chunk)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points_per_chunk)
        self.num_points_per_chunk = num_points_per_chunk
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points_per_chunk)
            return torch.cat([x, pointfeat], 1), trans

