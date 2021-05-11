from __future__ import print_function, division
import torch.nn as nn
import numpy as np
import pickle 
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim.lr_scheduler import StepLR
import tqdm
from tqdm.notebook import tqdm
import torch, os
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

workers = 2
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5
window = 7
ngpu = 1
Batch_size = 128
criterion = torch.nn.BCELoss()


def gradients_cal(image):
    
    x_dir = torch.Tensor([[[[1, 0.5, -1],
                        [2, 0.5, -2],
                        [1, 0.5, -1]]]]).to(device)
    y_dir = torch.Tensor([[[[1, 2, 1],
                        [0.5, 0.5, 0.5],
                        [-1, -2, -1]]]]).to(device)

    x_dir = x_dir.repeat(1,3,1,1)  
    y_dir = y_dir.repeat(1,3,1,1) 
                  
    grad_x = F.conv2d(image, x_dir, padding=1)                   
    grad_y = F.conv2d(image, y_dir, padding=1)
    return grad_x.to(device), grad_y.to(device)

def run_epoch(i, train_loader, G, D, Batch_size):
    loss_acc_g = 0
    loss_acc_d = 0
    loss_acc = 0
    count = 0
    print("Finding Z optimal")
    loss_d = 0
    for data in train_loader:
      zopt = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, (Batch_size,100,1,1))).to(device))
      optimizer = torch.optim.Adam([zopt],lr=lr,betas=(beta1, 0.999))
      image, masked_image, mask, weight = data
      for jj in range(1500):
        if(jj%100==0 and jj>0):
          print("Iteration = ",jj," Loss = ", loss_d)
        
        fake_samples = G(torch.clamp(zopt,min = -1,max = 1))
        prediction_on_fake = D(fake_samples)  
        context_loss = torch.norm(torch.mul(weight.to(device), fake_samples.to(device) - masked_image.to(device)),p=1)/Batch_size

        loss_fake = criterion(prediction_on_fake.squeeze(),torch.ones(prediction_on_fake.shape[0]).to(device))
        loss_d = context_loss + 0.003*loss_fake
        loss_d.backward()
        optimizer.step()
    

      feku = G(zopt)
      for mm in range(10,50):
        
        gg = feku[mm].clone()
        maskim = mask[mm]

        gg = gg*(1-maskim).to(device) + image[mm].to(device) * (maskim.to(device))

        save_image(gg, '/content/drive/MyDrive/MIC/lab07_170070021/save2/output_wp/{0}.png'.format(mm))
        save_image(masked_image[mm], '/content/drive/MyDrive/MIC/lab07_170070021/save2/input_images/{0}.png'.format(mm))
        save_image(mask[mm], '/content/drive/MyDrive/MIC/lab07_170070021/save2/input_images/mask{0}.png'.format(mm))

      print("Poisson optmisation")
      mask = mask.detach()
      print(mask.size())
      feku =  feku.detach()
      masked_image = masked_image.detach()
      start = masked_image.detach().to(device) + (1-mask.detach().to(device))*feku.detach().to(device)
      poisson_opt = nn.Parameter(torch.FloatTensor(start.detach().cpu().numpy()).to(device))
      opti_poisson = torch.optim.Adam([poisson_opt])
      fekx,feky = gradients_cal(feku.to(device))
      for jj in range(500):
          opti_poisson.zero_grad()
          poisson_opt_x, poisson_opt_y = gradients_cal(poisson_opt.to(device))
          poisson_loss = torch.sum(((fekx-poisson_opt_x)**2 + (fekx-poisson_opt_y)**2)*(1-mask.to(device)))
          poisson_loss.backward()
          poisson_opt.grad = poisson_opt.grad*(1-mask.to(device))
          opti_poisson.step()
          if(jj%100==0 and jj>0):
            print("Iteration = ", jj," Loss = ", poisson_loss)

      poisson_opt =  poisson_opt.detach()
      for mm in range(10,50):      
        gg = poisson_opt[mm].clone()
        save_image(gg, '/content/drive/MyDrive/MIC/lab07_170070021/save2/finaloutput/{0}.png'.format(mm)) 
      count = count + 1  
      print(count)
      if(count==1):
        break

    return 
class celeb(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        celeb_image = self.data[idx][0]
        coords1= np.random.randint(low = 20, high = 32, size=2)
        coords2= np.random.randint(low = 32, high = 44, size=2)
        masked_image = celeb_image.clone()
        masked_image[:,coords1[0]:coords2[0],coords1[1]:coords2[1]] = 0
        mask = torch.ones(celeb_image.shape)
        mask[:,coords1[0]:coords2[0],coords1[1]:coords2[1]] = 0
        mask2d = torch.unsqueeze(torch.unsqueeze(mask[0],0),0)
        conv_filter = torch.ones((1,1,window,window))
        conv_filter[0,0,int(window/2),int(window/2)] = 0
        weight = F.conv2d((1-mask2d), conv_filter, padding=int(window/2))/(window*window-1)
        
        weight = weight.squeeze()
        weight[coords1[0]:coords2[0],coords1[1]:coords2[1]] = 0
      
        weight = weight.repeat(3,1,1)

        return celeb_image, masked_image, mask, weight
                                

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

     
    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
       


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator(ngpu).to(device)
    D = Discriminator(ngpu).to(device)
    gmodel =torch.load("/content/drive/MyDrive/MIC/lab07_170070021/best_models2/model_bestg.zip")
    dmodel = torch.load("/content/drive/MyDrive/MIC/lab07_170070021/best_models2/model_bestd.zip")
    G.load_state_dict(gmodel)
    D.load_state_dict(dmodel)
        
   
    IMAGE_PATH = "/content/dats/"
    image_size = 64

    
    transform = transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    ind = list(range(0, 20000))                           
    dat = ImageFolder(IMAGE_PATH, transform)
    
    small_dat = torch.utils.data.Subset(dat, ind)

    Batch_size = 128
    dataset = celeb(small_dat)  
    train_loader = torch.utils.data.DataLoader(
                  dataset, batch_size = Batch_size,
                  shuffle=False,
                  drop_last=True)

    best_loss = 100000
    num_epochs = 1
    for i in range(num_epochs):

      run_epoch(i, train_loader, G, D, Batch_size)
          

if __name__ == "__main__":
    main()