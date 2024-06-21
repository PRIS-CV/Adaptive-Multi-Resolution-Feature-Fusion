from __future__ import print_function
import os
import time
import torch
import logging
import argparse
import torchvision
import torch.nn as nn
import numpy as np
import csv
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

use_cuda = torch.cuda.is_available()
criterion = nn.CrossEntropyLoss()

def dataset(dataset_path, resize=550, crop=448):
    '''
        -resize: the size of resize, defult=550
        -crop: the size of crop, defult=448
    '''
       
    transform_train = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    trainset    = torchvision.datasets.ImageFolder(root=dataset_path + '/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=16, drop_last = True)

    testset = torchvision.datasets.ImageFolder(root=dataset_path + '/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=16, drop_last = True)

    return trainloader, testloader

def cosine_anneal_schedule(t, lr, nb_epoch=100):
    cos_inner = np.pi * (t % (nb_epoch  ))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( lr/2* cos_out)

def train(epoch, net, trainloader,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct_low = 0
    correct_high = 0
    correct_fusion = 0
    total = 0
    idx = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        output = net(inputs)

        output_low = output[0]
        output_high = output[1]
        output_fusion = output[2]

        loss_low = criterion(output_low, targets)
        loss_high = criterion(output_high, targets)
        loss_fusion = criterion(output_fusion, targets)

        loss = loss_low + loss_high + loss_fusion
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted_low = torch.max(output_low.data, 1)
        _, predicted_high = torch.max(output_high.data, 1)
        _, predicted_fusion =  torch.max(output_fusion.data, 1)

        total += targets.size(0)

        correct_low += predicted_low.eq(targets.data).cpu().sum().item()
        correct_high += predicted_high.eq(targets.data).cpu().sum().item()
        correct_fusion += predicted_fusion.eq(targets.data).cpu().sum().item()

    train_acc_low = 100.*correct_low/total
    train_acc_high = 100.*correct_high/total
    train_acc_fusion = 100.*correct_fusion/total
    train_loss = train_loss/(idx+1)

    return train_acc_low, train_acc_high, train_acc_fusion, train_loss

def train_maml(epoch, net, trainloader,optimizer, optimizer_maml, maml_model):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct_low = 0
    correct_high = 0
    correct_fusion = 0
    total = 0
    idx = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        output = net(inputs)
        optimizer.zero_grad()

        #trival training
        output_low = output[0]
        output_high = output[1]
        output_fusion = output[2]

        loss_low = criterion(output_low, targets)
        loss_high = criterion(output_high, targets)
        loss_fusion = criterion(output_fusion, targets)

        loss = loss_low + loss_high + loss_fusion
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        #maml training
        optimizer_maml.zero_grad()
        task_model = maml_model.clone()
        output_1 = task_model(inputs)
        adaptation_loss = criterion(output_1[0], targets)  #loss_low respond to outerloop
        task_model.adapt(adaptation_loss)
        output_2 = task_model(inputs)
        evluation_loss = criterion(output_2[2], targets) / criterion(output_2[1], targets) #loss_fusion respond to innerloop
        evluation_loss.backward()
        optimizer_maml.step()


        _, predicted_low = torch.max(output_low.data, 1)
        _, predicted_high = torch.max(output_high.data, 1)
        _, predicted_fusion =  torch.max(output_fusion.data, 1)

        total += targets.size(0)

        correct_low += predicted_low.eq(targets.data).cpu().sum().item()
        correct_high += predicted_high.eq(targets.data).cpu().sum().item()
        correct_fusion += predicted_fusion.eq(targets.data).cpu().sum().item()

    train_acc_low = 100.*correct_low/total
    train_acc_high = 100.*correct_high/total
    train_acc_fusion = 100.*correct_fusion/total
    train_loss = train_loss/(idx+1)

    return train_acc_low, train_acc_high, train_acc_fusion, train_loss

def test(epoch, net, testloader):

    net.eval()
    test_loss = 0
    correct_low = 0
    correct_high = 0
    correct_fusion = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            output = net(inputs)

            output_low = output[0]
            output_high = output[1]
            output_fusion = output[2]

            loss_low = criterion(output_low, targets)
            loss_high = criterion(output_high, targets)
            loss_fusion = criterion(output_fusion, targets)

            loss = loss_low + loss_high + loss_fusion
            
            test_loss += loss.item()

            _, predicted_low = torch.max(output_low.data, 1)
            _, predicted_high = torch.max(output_high.data, 1)
            _, predicted_fusion =  torch.max(output_fusion.data, 1)
            
            total += targets.size(0)

            correct_low += predicted_low.eq(targets.data).cpu().sum().item()
            correct_high += predicted_high.eq(targets.data).cpu().sum().item()
            correct_fusion += predicted_fusion.eq(targets.data).cpu().sum().item()

    test_acc_low = 100.*correct_low/total
    test_acc_high = 100.*correct_high/total
    test_acc_fusion = 100.*correct_fusion/total
    test_loss = test_loss/(idx+1)

    return test_acc_low, test_acc_high, test_acc_fusion, test_loss

