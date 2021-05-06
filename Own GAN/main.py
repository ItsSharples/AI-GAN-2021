# -*- coding: utf-8 -*-


import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS

import numpy as np
import os 
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import torchvision.utils as vutils

# The Input Images
resize = 200
size = 64
colour_channels = 1

###
batch_size = 128
###

num_epochs = 100
z_size = 100

# How many Features you want it to identify in each Network
feature_size_dis = 64
feature_size_gen = 64

# Rates
dis_learning_rate = 0.0004
gen_learning_rate = 0.0001
lr_after_75 = 0.00005 # Reduce the Learning Rate at the 'End' of the Game
beta_rate = 0.5

# Goal Prediction Probability
real_ = 1
not_real_ = 0.9 # I don't want it to be 100% sure of an image
fake_ = 0.

# Paths to Inputs/Outputs
dataroot = r"./Images"
basePath = r"./Output"


### Derived Values
num_pixels = colour_channels * size * size

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         #transforms.Normalize((.5, .5, .5), (.5, .5, .5))
         transforms.Normalize((.5,), (.5,))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)# Load data

def own_data():
    if colour_channels == 1:
        compose = transforms.Compose(
            [
             transforms.Resize(resize),
             transforms.RandomRotation(60),
             transforms.RandomAffine(60),
             transforms.Resize(int(size*1.5)),
             transforms.CenterCrop(size),
             transforms.ToTensor(),

             transforms.Grayscale(),
             transforms.Normalize((.5,), (.5,)),
             
            ])  
    if colour_channels == 3:
        compose = transforms.Compose(
            [
             transforms.Resize(resize),
             transforms.RandomRotation(60),
             transforms.RandomAffine(60),
             transforms.Resize(int(size*1.5)),
             transforms.CenterCrop(size),
             transforms.ToTensor(),
             transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ])

    return ImageFolder(dataroot, compose)# Load data


#data = mnist_data()# Create loader with data, so that we can iterate over it

data = own_data()
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
# Num batches
#num_batches = len(data_loader)


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        
        hidden16size = feature_size_dis * 16
        hidden8size = feature_size_dis * 8
        hidden4size = feature_size_dis * 4
        hidden2size = feature_size_dis * 2
        hidden1size = feature_size_dis * 1
        
        self.hidden1 = nn.Sequential( 
             nn.Conv2d(colour_channels, hidden1size, 4, stride=2, padding=1, bias=False), 
             nn.LeakyReLU(0.2, inplace=True)
             )
        self.hidden2 = nn.Sequential( 
             nn.Conv2d(hidden1size, hidden2size, 4, stride=2, padding=1, bias=False), 
             nn.BatchNorm2d(hidden2size),
             nn.LeakyReLU(0.2, inplace=True),
             )
        self.hidden4 = nn.Sequential( 
             nn.Conv2d(hidden2size, hidden4size, 4, stride=2, padding=1, bias=False), 
             nn.BatchNorm2d(hidden4size),
             nn.LeakyReLU(0.2, inplace=True),
             )
        self.hidden8 = nn.Sequential( 
             nn.Conv2d(hidden4size, hidden8size, 4, stride=2, padding=1, bias=False), 
             nn.BatchNorm2d(hidden8size),
             nn.LeakyReLU(0.2, inplace=True),
             )
        
        self.out8 = nn.Sequential(
             nn.Conv2d(hidden8size, 1, 4, stride=1, padding=0, bias=False), 
             nn.Sigmoid()
             )
        
        
        self.hidden16 = nn.Sequential( 
             nn.Conv2d(hidden8size, hidden16size, 4, stride=2, padding=1, bias=False), 
             nn.BatchNorm2d(hidden16size),
             nn.LeakyReLU(0.2, inplace=True),
             )
        self.out16 = nn.Sequential(
             nn.Conv2d(hidden16size, 1, 4, stride=1, padding=0, bias=False), 
             nn.Sigmoid()
             )


    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden4(x)
        x = self.hidden8(x)
        #x = self.hidden16(x)
        x = self.out8(x)
        return x

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()        
        
        hidden16size = feature_size_gen * 16
        hidden8size = feature_size_gen * 8
        hidden4size = feature_size_gen * 4
        hidden2size = feature_size_gen * 2
        hidden1size = feature_size_gen * 1
        
        

        self.entry16 = nn.Sequential( 
            nn.ConvTranspose2d(z_size, hidden16size, 4, 1, 0, bias = False),
            nn.BatchNorm2d(hidden16size),
            nn.LeakyReLU(0.2)
        )
        self.hidden8 = nn.Sequential( 
            nn.ConvTranspose2d(hidden16size, hidden8size, 4, 2, 1, bias = False),
            nn.BatchNorm2d(hidden8size),
            nn.LeakyReLU(0.2)
        )
        self.entry8 = nn.Sequential( 
            nn.ConvTranspose2d(z_size, hidden8size, 4, 1, 0, bias = False),
            nn.BatchNorm2d(hidden8size),
            nn.LeakyReLU(0.2)
        )
        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(hidden8size, hidden4size, 4, 2, 1, bias = False),
            nn.BatchNorm2d(hidden4size),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(hidden4size, hidden2size, 4, 2, 1, bias = False),
            nn.BatchNorm2d(hidden2size),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(hidden2size, hidden1size, 4, 2, 1, bias = False),
            nn.BatchNorm2d(hidden1size),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(hidden1size, colour_channels, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, x):
        #x = self.hidden16(x)
        x = self.entry8(x)
        x = self.hidden4(x)
        x = self.hidden2(x)
        x = self.hidden1(x)
        x = self.out(x)
        return x
 
# def images_to_vectors(images):
#     return images.view(images.size(0), num_pixels)

# def vectors_to_images(vectors):
#     return vectors.view(vectors.size(0), colour_channels, size, size)

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    return torch.randn(size, z_size, 1, 1, device=device)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# def train_discriminator(optimizer, real_data, fake, label):
#     ## Train with all-real batch
#     discriminator.zero_grad()
#     # Format batch
#     #real_data = real_batch.to(device)
#     label.fill_(real_)
#     # Forward pass real batch through D
#     output = discriminator(real_data).view(-1)
#     # Calculate loss on all-real batch
#     errD_real = loss(output, label)
#     # Calculate gradients for D in backward pass
#     errD_real.backward()
#     D_x = output.mean().item()
    
#     ## Train with all-fake batch
    
#     # Generate a new Image
#     label.fill_(fake_)
#     # Does it think it's Fake?
#     output = discriminator(fake.detach()).view(-1)
#     # Calculate D's loss on the all-fake batch
#     errD_fake = loss(output, label)
#     # Calculate the gradients for this batch, accumulated (summed) with previous gradients
#     errD_fake.backward()
#     D_G_z1 = output.mean().item()
#     # Compute error of D as sum over the fake and the real batches
#     errD = errD_real + errD_fake
#     # Update D
#     d_optimizer.step()
    
#     return errD, D_x, D_G_z1

# def train_generator(optimizer, fake_data, label):
#     generator.zero_grad()
#     label.fill_(real_)  # fake labels are real for generator cost
#     # Since we just updated D, perform another forward pass of all-fake batch through D
#     output = discriminator(fake_data).view(-1)
#     # Calculate G's loss based on this output
#     errG = loss(output, label)
#     # Calculate gradients for G
#     errG.backward()
#     D_G_z2 = output.mean().item()
#     # Update G
#     optimizer.step()
#     return errG

      
device = "cuda:0"  


discriminator = DiscriminatorNet().to(device)
generator = GeneratorNet().to(device)
      
generator.apply(weights_init)
discriminator.apply(weights_init)


d_optimizer = optim.Adam(discriminator.parameters(), lr=dis_learning_rate, betas=(beta_rate, 0.999))
g_optimizer = optim.Adam(    generator.parameters(), lr=gen_learning_rate, betas=(beta_rate, 0.999))


loss = nn.BCELoss()


num_test_samples = 64
test_noise = noise(num_test_samples)


# =============================================================================
# def imshow(inp, title=None):
#     inp = inp.cpu()
#     """Imshow for Tensor."""
#     inp = inp.numpy().reshape(-1, 28, 28)
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
# =============================================================================

# Track Progress
img_list = []
G_losses = []
D_losses = []
iters = 0
current_epoch = 0


def normalise(image):
    # Normalised [0,1]
    return (image - np.min(image))/np.ptp(image)

def drawOriginals():
    # Plot some training images
    output_dir = f"{basePath}/Original Images"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    images = (next(iter(data_loader))[0].to(device)[:64]).cpu().numpy().transpose(0, 2, 3, 1)
    
    for image in range(len(images)):
        plt.figure(0)
        plt.clf()
        plt.axis('off')    
        imshow(normalise(images[image]))
        #plt.title(f"Generated Epoch {epoch}")
        plt.savefig(f"{output_dir}/Generated_Image_{image}.png", format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
    


def trainModels():
    final_images = {}
    for raw_epoch in range(num_epochs):
        epoch = raw_epoch + current_epoch
        print(f"Epoch No: {epoch}")
        
        if epoch == 75:
            d_optimizer.learning_rate = lr_after_75
            g_optimizer.learning_rate = lr_after_75
        for n_batch, (real_batch,_) in enumerate(data_loader):
            real_data = real_batch.to(device)
            # Configure the Data
            size_ = real_data.size(0)
            # Generate Fake Images
            fake_data = generator(noise(size_))
            label = torch.full((size_,), real_, dtype=torch.float, device=device)
            
            
            # Train Discriminator
            #d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data, label)
            # Train Generator
            #g_error = train_generator(g_optimizer, fake_data, label)
            
            ### Train the Discriminator
            
            ## Train with all-real batch
            
            # Zero out the Gradiants
            discriminator.zero_grad()
            # Aim for Guessing it's Real 
            label.fill_(not_real_)
            # Does it think it's Real?
            output = discriminator(real_data).view(-1)
            # Calculate loss on all-real batch
            errD_real = loss(output, label)
            # Calculate gradients for this batch, summed with previous gradients
            errD_real.backward()
            D_x = output.mean().item()
            
            ## Train with all-fake batch
            
            # Aim for Guessing it's Fake 
            label.fill_(fake_)
            # Does it think it's Fake?
            output = discriminator(fake_data.detach()).view(-1)
            # Calculate loss on all-fake batch
            errD_fake = loss(output, label)
            # Calculate gradients for this batch, summed with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            d_optimizer.step()
            
            ### Train the Generator
            
            generator.zero_grad()
            label.fill_(real_)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake_data).view(-1)
            # Calculate G's loss based on this output
            errG = loss(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            g_optimizer.step()
            
            d_error = errD
            g_error = errG
    
            # if d_error < best_test:
            #     best_test = d_error
            #     test_images = vectors_to_images(d_pred_fake)
            #     test_images = test_images.data
            #     images = test_images.cpu().numpy().transpose(0, 2, 3, 1)
            #     imshow(images[0])
            #     plt.title(f"Best for Epoch {epoch}")
            
            # Save Losses for plotting later
            G_losses.append(g_error.item())
            D_losses.append(d_error.item())
    
        with torch.no_grad():
            test_images = generator(test_noise).detach().cpu()
            images = test_images.numpy().transpose(0, 2, 3, 1)
            output_dir = f"{basePath}/Epoch {epoch}"
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            if epoch % 5 == 0:
                torch.save(generator, '/'.join([output_dir, "checkpoint_generator.pth"]))
                torch.save(discriminator, '/'.join([output_dir, "checkpoint_discriminator.pth"]))
            for image in range(len(images)):
                final_images[f"{output_dir}/Generated_Image_{image}.png"] = images[image]
                
        
        if epoch % 10 == 0:
            for location in final_images.keys():
                plt.figure(0)
                plt.clf()
                plt.axis('off')
                imshow(normalise(final_images[location]))
                #plt.title(f"Generated Epoch {epoch}")
                plt.savefig(location, format="png", bbox_inches="tight", pad_inches=0)
                plt.close()
            final_images = {}
    
                
    for location in final_images.keys():
        plt.figure(0)
        plt.clf()
        plt.axis('off')
        imshow(normalise(final_images[location]))
        #plt.title(f"Generated Epoch {epoch}")
        plt.savefig(location, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        
def loadModels(epoch = 90):
    global current_epoch, d_optimizer, g_optimizer
    output_dir = f"{basePath}/Epoch {epoch}"
    tmp_gen = torch.load('/'.join([output_dir, "checkpoint_generator.pth"]))
    tmp_dis = torch.load('/'.join([output_dir, "checkpoint_discriminator.pth"]))
    
    tmp_gen.apply(weights_init)
    tmp_dis.apply(weights_init)   

    d_optimizer = optim.Adam(tmp_gen.parameters(), lr=dis_learning_rate, betas=(beta_rate, 0.999))
    g_optimizer = optim.Adam(tmp_dis.parameters(), lr=gen_learning_rate, betas=(beta_rate, 0.999))
    current_epoch = epoch
    return tmp_gen, tmp_dis
        
def generateImage(epoch = 95, allOfThem = False, toFile = False):
    output_dir = f"{basePath}/Epoch {epoch}"
    tmp_gen = torch.load('/'.join([output_dir, "checkpoint_generator.pth"]))
    size = next(iter(data_loader))[0].size(0)
    images = tmp_gen(noise(size)).detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    if allOfThem:
        for image in range(len(images)):
            plt.figure()
            plt.clf()
            plt.axis('off')
            imshow(normalise(images[image]))
            if toFile:
                plt.savefig(f"{output_dir}/Generated_Image_{image}.png", format="png", bbox_inches="tight", pad_inches=0)
                plt.close()
    else:
        plt.figure()
        plt.clf()
        plt.axis('off')
        imshow(normalise(images[0]))
        if toFile:
            plt.savefig(f"{output_dir}/Generated_Image_0.png", format="png", bbox_inches="tight", pad_inches=0)
            plt.close()
    
    
        
    

        #plt.title(f"Generated Epoch {epoch}")

    
#generateImage()

#num_epochs = 15
#generator, discriminator = loadModels(90)
#generateImage(100, True, True)
trainModels()


#for epoch in [25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400]:
#    generator, discriminator = loadModels(epoch)
#    generateImage()
