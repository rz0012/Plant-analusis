import os
import numpy as np
from tqdm import tqdm
import imghdr 
import torchvision
from torchvision import transforms 
from torchvision.transforms import Compose 
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pdb
import torch.optim as optim 
import time as time
import pandas as pd
import PIL
from PIL import Image
import pickle
from torch.utils.data import DataLoader
from dataloader import dataset
from preprocessing import transform
from model_init import init_model


if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
torch.backends.cudnn.benchmark = True


#creating model using pretrained model
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(512, 3)
model = nn.DataParallel(model)
model = model.to(device) 


def eval(X,y,outputs,train = False):
    
    correct = 0.0
    total = 0.0
    prediction = torch.argmax(outputs, 1)
    for i,j in zip(prediction,y):
      if i.item() == j.item():
        correct += 1
      total += 1 
    return (correct/total)


def Train(data_dir,EPOCH ,BATCH_SIZE,SIZE,LR,MOM):
    partition = pickle.load(open(os.path.join(data_dir, 'partition.p'), 'rb'))
    label = pickle.load(open(os.path.join(data_dir, 'label.p'), 'rb'))

    training_dataset = dataset(partition['train'], label, data_dir, transform=transform)
    training_generator = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = dataset(partition['test'], label, data_dir, transform=transform)
    test_generator = DataLoader(test_dataset, batch_size=SIZE, shuffle=True)


    
    

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM)
    data = []
    for epoch in range(EPOCH):
        Losss = []
        acc = []
        test_loss = []
        test_acc = []

        model.train()
        for batch_idx, batch_info in tqdm(enumerate(training_generator)):
            batch_data, batch_labels = batch_info[0].to(device), batch_info[1].to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = loss_function(outputs, batch_labels)
            Losss.append(loss.item())
            loss.backward()
            optimizer.step()

            acc.append(eval(batch_data,batch_labels,outputs,train = False))

            
            
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_info in tqdm(enumerate(test_generator)):
                batch_data, batch_labels = batch_info[0].to(device), batch_info[1].to(device)
                outputs = model(batch_data)
                loss = loss_function(outputs, batch_labels)
                test_loss.append(loss)
                test_acc.append(eval(batch_data,batch_labels,outputs,train = False))

        train_acc = float(sum(acc)/len(acc))*100
        train_loss = float(sum(Losss)/len(Losss))
        Val_Acc = float(sum(test_acc)/len(test_acc))*100
        Val_Loss = float(sum(test_loss)/len(test_loss))


        print(f"epoch = {epoch+1}   Acc = {train_acc}   Loss = {train_loss}   val_acc = {Val_Acc}   val_loss = {Val_Loss}")
        data.append([epoch+1, train_acc, train_loss, Val_Acc, Val_Loss])

    torch.save(model, '/data/plant_domain_classification/dataset/server_task_2/model_resnet34_25_0.01_0.01.pth')
      
    return data



data_dir = input("please input location where your data is: ")
data = Train(data_dir,25,128,128,0.01,0.1)

data = pd.DataFrame(data,columns=["Epoch", "Train Accuracy", "Train Loss", "Test Accuracy", "Test Loss"])

print()
print(data)

data.to_csv("/data/plant_domain_classification/dataset/server_task_2/plant_trained_data_25_0.01_0.01.csv")


