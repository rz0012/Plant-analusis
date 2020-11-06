import PIL
from PIL import Image
import torchvision
from torchvision import transforms 
import torch
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import transform

if torch.cuda.is_available():
        device = torch.device("cuda:0")  
        print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")



mname = input("please input model location: ")


net = torch.load(mname).to(device)

while(1):
        

    fname = input("please enter image location: ")
    

    image = Image.open(fname)

    plt.imshow(image)

    image = transform(image)

    image = image.view(2,224,224)

    net.eval()


    pred = net(image.to(device))

    pred = torch.argmax(pred,1)
    

    if pred == 1:
        print("leaf/leaves")
    elif pred == 2:
        print("tree")
    elif pred == 0:
        print("branch")

    end = input("do you want to predict another? Y: yes, N: no >>")
    if end.lower() == n:
        break

