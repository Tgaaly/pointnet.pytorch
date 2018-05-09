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
from loss import FocalLoss
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--lr', type=float, default=0.001, help='input batch size')
parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls_1024_b64_300epoch_reduceLR50',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--focal_loss', action='store_true')
parser.add_argument('--transform_reg', action='store_true')
opt = parser.parse_args()
print (opt)

if opt.focal_loss:
    opt.outf += '_fl'

if opt.transform_reg:
    opt.outf = '_ce_reg'

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = 12345#random.randint(1, 10000) # fix seed
# print("Random Seed: ", opt.manualSeed)
print("Set Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

data_dir = '/home/tarek/Data/ModelNet40/ModelNet40'

dataset = Modelnet40(root = data_dir, classification = True, npoints = opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = Modelnet40(root = data_dir, classification = True, train = False, npoints = opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


classifier = PointNetCls(k = num_classes, num_points = opt.num_points)


if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = opt.lr * (0.8 ** (epoch // 20))
    lr = opt.lr * (0.8 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print 'learning rate = ', param_group['lr']

# optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)

classifier.cuda()

num_batch = len(dataset)/opt.batchSize

best_train_acc = 0
best_test_acc = 0
all_losses = []

if opt.focal_loss:
    gamma=2
    alpha=0.25
    criterion_focal_loss = FocalLoss(gamma)

f = open('/home/tarek/Dropbox/desktop_results/results_morepoints.txt','w')

for epoch in range(opt.nepoch):
    
    # Adjust learning rate.
    adjust_learning_rate(optimizer, epoch)

    all_correct = 0
    all_num = 0
    classifier.train()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points, target = Variable(points), Variable(target[:,0])
        points = points.transpose(2,1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        pred, trans3, trans64 = classifier(points)

        if opt.focal_loss:
            pred = F.softmax(pred)
            loss = criterion_focal_loss(pred.cpu(), target.cpu())
        elif opt.transform_reg:
            pred = F.log_softmax(pred)
            loss = F.nll_loss(pred, target)

            # Regularization weight.
            kappa = 0.001

            # # Regularization of input transform.
            # iden3 = np.eye(trans3.size(1))
            # iden3 = np.tile(iden3,(trans3.size(0),1,1))
            # iden3 = torch.Tensor(iden3)
            # iden3 =  torch.autograd.Variable(iden3, requires_grad=False).cuda()#requires_grad is False by default
            # mat = torch.bmm(trans3, torch.transpose(trans3,1,2))
            # reg3 = F.mse_loss(mat, iden3)
            # loss += kappa*reg3

            # Regularization of feature transform.
            iden64 = np.eye(trans64.size(1))
            iden64 = np.tile(iden64,(trans64.size(0),1,1))
            iden64 = torch.Tensor(iden64)
            iden64 =  torch.autograd.Variable(iden64, requires_grad=False).cuda()#requires_grad is False by default
            mat = torch.bmm(trans64, torch.transpose(trans64,1,2))
            reg64 = F.mse_loss(mat, iden64)
            loss += kappa*reg64
        else:            
            pred = F.log_softmax(pred)
            loss = F.nll_loss(pred, target)        

        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        all_correct += correct
        all_num += int(target.data.size(0))
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.data[0], correct/float(int(target.data.size(0)))))
        all_losses.append(loss.data[0])
        np.save('all_losses', all_losses)

    if (all_correct/float(all_num)) > best_train_acc:
        best_train_acc = all_correct/float(all_num)

    # Save model.
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    f.write('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.data[0], correct/float(int(target.data.size(0)))) + '\n')

    # Run on test set (end of every epoch). 
    classifier.eval()
    correct = 0
    num = 0
    for i, data in enumerate(testdataloader, 0):
        points, target = data
        points, target = Variable(points), Variable(target[:,0])
        points = points.transpose(2,1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        pred, _, _ = classifier(points)

        if opt.focal_loss:
            pred = F.softmax(pred)
        else:
            pred = F.log_softmax(pred)

        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(target.data).cpu().sum()
        num += int(target.data.size(0))

    if (correct/float(num)) > best_test_acc:
        best_test_acc = correct/float(num)
        print 'updating best test accuracy...'
        f.write('updating best test accuracy...' + '\n')
    print 'best_test_acc: ', best_test_acc
f.write('best_test_acc: '+ str(best_test_acc) + '\n')




print 'best_train_acc: ', best_train_acc
print 'best_test_acc: ', best_test_acc
f.write('best_train_acc: '+ str(best_train_acc) + '\n')
f.write('best_test_acc: '+ str(best_test_acc) + '\n')
np.save('all_losses', all_losses)

f.close()
