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

resize = 200
size = 128
colour_channels = 1

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
            [transforms.ToTensor(),
             transforms.RandomRotation(15),
             transforms.RandomAffine(15),
             #transforms.Resize(224),
             
             #transforms.ColorJitter(0.1, 0.1, 0.1),
             #transforms.Normalize((.5, .5, .5), (.5, .5, .5))
             
             transforms.Resize((resize,resize)),
             transforms.CenterCrop(size),
             
             transforms.Grayscale(),
             transforms.Normalize((.5,), (.5,)),
             
            ])  
    if colour_channels == 3:
        compose = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize((.5, .5, .5), (.5, .5, .5))
             transforms.Resize((size,size)),
             transforms.Normalize((.5,), (.5,))
            ])

    location = r"./Images"
    
    image_datasets = ImageFolder(location, compose)
    
    return image_datasets# Load data



data = mnist_data()# Create loader with data, so that we can iterate over it

data = own_data()




data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)


num_pixels = colour_channels * size * size

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = num_pixels
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
discriminator = DiscriminatorNet()
    
    
    
def images_to_vectors(images):
    return images.view(images.size(0), num_pixels)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), colour_channels, size, size)


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = num_pixels
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    

generator = GeneratorNet()



def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    n = n.cuda()
    return n


generator.cuda()
discriminator.cuda()

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)



loss = nn.BCELoss()

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    data = data.cuda()
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    data = data.cuda()
    return data



def train_discriminator(optimizer, real_data, fake_data):
    real_data, fake_data = real_data.cuda(), fake_data.cuda();
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    N = fake_data.size(0)    # Reset gradients
    optimizer.zero_grad()    # Sample noise and generate fake data
    prediction = discriminator(fake_data)    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()    # Update weights with gradients
    optimizer.step()    # Return error
    return error



num_test_samples = 16
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

basePath = "Output"
# Total number of epochs to train
num_epochs = 500
for epoch in range(num_epochs):
    print(f"Epoch No: {epoch}")
    plt.pause(0.0000001)
    for n_batch, (real_batch,_) in enumerate(data_loader):
        real_batch = real_batch.cuda()
        
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch)).cuda()
        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).cuda().detach() # Train D
        
        #real_data, fake_data = real_data.cuda(), fake_data.cuda()
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log batch error
        #logger.log(d_error, g_error, epoch, n_batch, num_batches)
        
        # if d_error < best_test:
        #     best_test = d_error
        #     test_images = vectors_to_images(d_pred_fake)
        #     test_images = test_images.data
        #     images = test_images.cpu().numpy().transpose(0, 2, 3, 1)
        #     imshow(images[0])
        #     plt.title(f"Best for Epoch {epoch}")
                
            
        
        # Display Progress every few batches
        #if (n_batch) % 100 == 0:
            
    test_images = vectors_to_images(generator(test_noise))
    test_images = test_images.data
    images = test_images.cpu().numpy().transpose(0, 2, 3, 1)
    output_dir = f"{basePath}/Epoch {epoch}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for image in range(len(images)):
        plt.figure(0)
        plt.clf()
        plt.axis('off')
        imshow(images[image])
        #plt.title(f"Generated Epoch {epoch}")
        plt.savefig(f"{output_dir}/Generated_Image_{image}.png", format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
            
    
    # real_images = vectors_to_images(real_data)
    # real_images = real_images.data
    # images = real_images.cpu().numpy().transpose(0, 2, 3, 1)
    # for image in range(len(images)):
    #     plt.figure(1)
    #     plt.axis('off')
    #     imshow(images[image])
    #     plt.savefig(f"{output_dir}/Original_Image_{image}.png", format="png", bbox_inches="tight", pad_inches=0)
    #     plt.close()
                
            #print(f"Disc Error {d_error}")
            #print(f"Gen Error: {g_error}")
            
            #logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            #logger.display_status(epoch, num_epochs, n_batch, num_batches,d_error, g_error, d_pred_real, d_pred_fake)