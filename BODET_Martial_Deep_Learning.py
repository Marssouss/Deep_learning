#!/usr/bin/env python3
# coding: utf-8

import time
start = time.time()

import sys 
import os
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

def read_folder(path):
    files = os.listdir(path)
    for name in files:
        if name.find(' ') != -1:
            os.rename(path+'/' + name, path+ '/' +name.replace(' ', '_'))

path_train  = './fruits-360/Training'
path_test = './fruits-360/Test'

read_folder(path_train)
read_folder(path_test)

train_dataset = datasets.ImageFolder(path_train, transform = transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)

test_dataset = datasets.ImageFolder(path_test, transform = transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = True)

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        # 1 input image channel, 16 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 40, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(40 * 22 * 22, 300) # (size of input, size of output)
        self.fc2 = nn.Linear(300, 84)
        self.fc3 = nn.Linear(84, 80)



    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#check CPU on machine 

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(device)

net = Net().to(device) # for cuda
#net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

epochs = 5
for epoch in range(epochs):
    running_loss=0.0
    for i, data in enumerate(train_loader,0):
        inputs,labels = data
        inputs,labels = inputs.to(device), labels.to(device) # for cuda
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if i % 2000 == 1999: #printevery2000mini-batches
            print ('[%d,%5d] loss: %.3f' % (epoch+1,i+1,running_loss/2000))
            running_loss=0.0
print ('Finished Training')

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))

testiter = iter(test_loader)
images, labels =  testiter.next()
imshow(torchvision.utils.make_grid(images))
print (labels)

images = images.to(device, torch.float)
outputs = net(images)
_,predicted = torch.max(outputs,1)
print (predicted)

correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data # cuda ?
        images = images.to(device, torch.float)
        labels = labels.to(device, torch.long)
        outputs=net(images)
        _,predicted= torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()   
print ('Training dataset : Accuracy of the network on the %(tot)s test images %(percent)s %%' % {'tot':total, 'percent':((100*correct)/int(total))})

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data # cuda ?
        images = images.to(device, torch.float)
        labels = labels.to(device, torch.long)
        outputs=net(images)
        _,predicted= torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()   
print ('Testing dataset : Accuracy of the network on the %(tot)s test images %(percent)s %%' % {'tot':total, 'percent':((100*correct)/int(total))})

#Time counter
print ("Time elapsed :", (time.time() - start)/60, "m")

