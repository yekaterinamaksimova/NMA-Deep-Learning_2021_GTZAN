# @title Define and load Data Loaders
import os
import glob
# import imageio
import random, shutil
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
#import librosa
#import librosa.display
import requests
# import cv2

import torch.nn.functional as F
from torchvision.utils import save_image
from sklearn.metrics import r2_score 
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from google.colab import drive

class SpectogramDataset(Dataset):
    def __init__(self, audio_labels, spectograms, train=False, valid=False):
        np.random.seed(0)
        nb_samples = len(audio_labels)
        permutation = np.random.permutation(nb_samples)
        if train:
            indices = permutation[:int(nb_samples *.6)+1]
        elif valid:
            indices = permutation[int(nb_samples *.6)+1: -1*int(nb_samples *.2)]
        else:
            indices = permutation[-1*int(nb_samples *.2):]
        self.audio_labels = audio_labels[indices]
        self.spectograms = spectograms[indices]
        self.max = np.max(np.max(self.spectograms, axis=-1), axis=-1)
        self.min = np.min(np.min(self.spectograms, axis=-1), axis=-1)

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        spect = self.spectograms[idx][:256,:640].astype(np.float32)
        spect = (spect - self.min[idx]) / (self.max[idx] - self.min[idx])
        # assert spect.max() == 1 and spect.min() == 0 
        label = self.audio_labels[idx]
        return spect[np.newaxis,...], label

#@title Alt dataloader, uses tensor instead of numpy, and includes  standardization option
#@title 
class TensorDataset(Dataset):
    def __init__(self, audio_labels, spectograms, train=False, valid=False, min_max_scaling=False, standardize=False):
        np.random.seed(0)
        self.min_max_scaling = min_max_scaling
        self.standardize = standardize
        nb_samples = len(audio_labels)
        permutation = np.random.permutation(nb_samples)
        if train:
            indices = permutation[:int(nb_samples *.6)+1]
        elif valid:
            indices = permutation[int(nb_samples *.6)+1: -1*int(nb_samples *.2)]
        else:
            indices = permutation[-1*int(nb_samples *.2):]
        self.audio_labels = audio_labels[indices]
        self.spectograms = torch.tensor(spectograms[indices], device=device, dtype=torch.float32)
      
    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        # Use within image min-max scaling/ standardization
        spect = self.spectograms[idx][:256,:640]
        if self.min_max_scaling:
            max = torch.max(self.spectograms)
            min = torch.min(self.spectograms)
            spect = (spect - min) / (max - min)
        if self.standardize:
            mean = torch.mean(self.spectograms)
            std = torch.std(self.spectograms)
            spect = (spect - mean) / std
            # assert spect.max() == 1 and spect.min() == 0 
        label = self.audio_labels[idx]
        return spect.unsqueeze(dim=0), label
    
    
# @title Load Data
def load_spectrograms(file_name):
    data_folder = '/nfs/scistore08/kondrgrp/emaksimo/GTZAN'
    spectrograms = np.load(os.path.join(data_folder, file_name))
    nb_genres, nb_songs, nb_freq, nb_tp = spectrograms.shape
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 
              'pop', 'reggae', 'rock']
    audio_labels = np.zeros((nb_genres, nb_songs))
    for i, genre in enumerate(genres):
        audio_labels[i: (i+1)*nb_songs, :] = i
    '''audio_labels = np.zeros((nb_genres, nb_songs, nb_genres))
    for i, genre in enumerate(genres):
      audio_labels[i, :, i] = 1'''
    # df_audio_labels = pd.DataFrame(audio_labels,columns=['Labels'])
    audio_labels = audio_labels.reshape((nb_genres*nb_songs))
    assert np.array_equal(audio_labels[0], audio_labels[10])
    spectrograms = spectrograms.reshape((nb_genres*nb_songs, nb_freq, nb_tp))
    return audio_labels, spectrograms, genres

def Set_Device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ',device)
    return device
"""
A Convolutional Variational Autoencoder
"""
class VAE_simple(nn.Module):
    def __init__(self, imgChannels=1, featureDim=16*40, zDim=100, batch_size = 5):
        super(VAE_simple, self).__init__()

        # self.batch_size = batch_size
        self.featureDim = featureDim

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder. out_H = H - kernel_size + 1 = H-1
        self.encConv1 = nn.Conv2d(in_channels=imgChannels, out_channels=16, kernel_size=3, stride = 2, padding=1) # 1, 256, 640 -> 16, 128, 320
        self.encConv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride = 2, padding=1) 

        self.dsConv12 = nn.MaxPool2d(2, stride=2) 
        self.dsConv23 = nn.MaxPool2d(2, stride=2) 

        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.usConv32 = nn.Upsample(scale_factor=4, mode='bicubic')
        self.usConv21 = nn.Upsample(scale_factor=4, mode='bicubic')
        self.decConv3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.decConv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.decConv1 = nn.Conv2d(in_channels=16, out_channels=imgChannels, kernel_size=3, padding=1)

        # Classification
        self.clFC1 = nn.Linear(zDim, 10)

    def encoder(self, x):

        x = self.dsConv12(F.relu(self.encConv1(x) )) # encConv1: 1, 256, 640 -> 16, 128, 320; dsConv12: 16, 64, 160
        x = self.dsConv23(F.relu(self.encConv2(x))) # encConv2:  16, 64, 160  -> 32, 32, 80  ; dsConv12: 32, 16, 40
        x = torch.mean(x, axis = 1) # averaging: 5, 32, 16, 40  -> 1, 16, 40 
        x = x.view(self.batch_size, self.featureDim)
        mu = self.encFC1(x) # Linear: 1, 16*40   ->  100
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        x = F.relu(self.decFC1(z)) # Linear:  100 -> 16*40
        x = x.view(self.batch_size, 1, 16, 40)
        x = self.decConv3(x) # decConv3: 1, 16, 40 -> 32, 16, 40 
        
        x = F.relu(self.decConv2(self.usConv32(x))) # usConv32: 32, 16, 40  -> 32, 64, 160 decConv2 -> 16, 64, 160 ##### usConv32: 32, 16, 20 -> 32, 128, 160, decConv2: -> 16, 128, 160
        # x = torch.sigmoid(self.decConv1(self.usConv21(x)))
        x = self.decConv1(self.usConv21(x)) # usConv21: 16, 64, 160 -> 16, 256,640 decConv2: -> 1, 256,640  ###### usConv21: 16, 128, 160 -> 16, 1024, 1280, decConv2: -> 1, 1024, 1280
        return x
    
    def classify(self, z):
        
        x = self.clFC1(z)
        # x = F.softmax(x) #torch.nn.Softmax(x)
        return x

    def forward(self, x):
        self.batch_size = x.shape[0]
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        class_out = self.classify(z)
        return out, mu, logVar, class_out

def train(model, device, train_loader, validation_loader, num_epochs, learning_rate, beta = 1):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    

    train_loss, train_class_loss = [], []
    train_acc, train_r2 = [], []
    
    with tqdm(range(num_epochs), unit='epoch') as tepochs:
        tepochs.set_description('Training')
        for epoch in tepochs:
            
            model.train()
            
            running_loss,running_KL_loss,running_class_loss,running_recon_loss,acc,r2_train,total = 0., 0., 0., 0., 0., 0., 0. 
            
            for idx, data in enumerate(train_loader, 0):

                imgs, labels = data
                imgs, labels = imgs.to(device), labels.to(device)

                output, mu, logVar, class_output = model(imgs)
                optimizer.zero_grad()
                class_loss = F.cross_entropy(class_output, labels.long(), reduction='sum')
                kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
                recon_loss = F.mse_loss(output, imgs, reduction='sum')
                loss = recon_loss + beta * kl_divergence + class_loss
                #print('KL Divergence', kl_divergence.item(), 'MSE', recon_loss.item(), 'class loss', class_loss.item())
                loss.backward()
                optimizer.step()

                tepochs.set_postfix(loss=loss.item())
                running_loss += loss  # add the loss for this batch
                running_KL_loss += kl_divergence
                running_class_loss += class_loss
                running_recon_loss += recon_loss

                pred_labels = torch.argmax(class_output, axis=-1)
                acc += (pred_labels.float() == labels).float().sum()
                # r2_train += np.sum(r2_score(torch.flatten(imgs, start_dim = 1).detach().numpy(), torch.flatten(output, start_dim = 1).detach().numpy(), multioutput = 'raw_values'))
                total += output.size(0)

                if idx%100 == 0:
                    print('Batch #{}. Current loss: {:.2f}, L_class {:.2f}%, L_recons {:.2f}% , L_KLdiv {:.2f}%. Accuracy train: {:.2f}'.format(idx, running_loss/total, 100*running_class_loss/running_loss, 100*beta*running_KL_loss/running_loss, 
                                  100*running_recon_loss/running_loss, 100*acc/total))

            # append the loss for this epoch
            train_loss.append(running_loss/len(train_loader))
            train_r2.append(r2_train/total)
            train_class_loss.append(running_class_loss/total)
            train_acc.append(100*acc/total)

            # evaluate on validation data
        
        # NO validation on every epoch
        validation_loss, validation_r2, validation_class_loss, validation_acc = model_evaluation(model, device, 
                                                                    validation_loader, learning_rate, beta = 1)

    return torch.stack(train_loss).cpu(), train_r2, validation_loss, validation_r2, torch.stack(train_class_loss).cpu(), torch.stack(validation_class_loss).cpu(), torch.stack(train_acc).cpu(), torch.stack(validation_acc).cpu()


def model_evaluation(model, device, validation_loader, learning_rate, beta = 1):

    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    
    validation_class_loss, validation_loss, validation_acc, validation_r2 = [], [], [], []
    
    running_loss, r2_val, total, running_class_loss, acc = 0.,0., 0., 0., 0.
#             running_recon_loss = 0.
    with torch.no_grad():
        for idx, data in enumerate(validation_loader, 0):
        # getting the validation set

            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            output, mu, logVar, class_output = model(imgs)
            optimizer.zero_grad()
            class_loss = F.cross_entropy(class_output, labels.long(), reduction='sum')
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            recon_loss = F.mse_loss(output, imgs, reduction='sum')
            loss = recon_loss + beta * kl_divergence + class_loss

#             tepochs.set_postfix(loss=loss.item())
            running_loss += loss.item()
            running_class_loss += class_loss
            # class_output_np = class_output.detach().numpy()
            # labels_np = labels.detach().numpy()
            pred_labels = torch.argmax(class_output, axis=-1)

            # get performance
            acc += (pred_labels == labels).float().sum() #np.sum(np.where(pred_labels == labels_np))
            running_class_loss += class_loss.item()
            # r2_val += np.sum(r2_score(torch.flatten(imgs, start_dim = 1).detach().numpy(), torch.flatten(output, start_dim = 1).detach().numpy(), multioutput = 'raw_values'))
            total += output.size(0)

        validation_loss.append(running_loss/len(validation_loader))
        validation_r2.append(r2_val/total)
        validation_class_loss.append(running_class_loss/total)
        validation_acc.append(100*acc/total)
    
    return validation_loss, validation_r2, validation_class_loss, validation_acc



def print_n_parameters(model):
    psum = 0
    for par in model.parameters():
        psum += torch.numel(par) 
    print('Number of parameters in the model: ', psum)
    return psum

  
def plot_loss(train_loss, validation_loss, train_acc, validation_acc, num_epochs, model, label = 'Validation'):
    model.eval()
    with torch.no_grad():
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,5))

        ax[0].set_title('Loss')
        ax[0].plot(np.arange(num_epochs), np.array(train_loss), '--o', label = 'Training')
        if len(validation_loss) == 1:
            ax[0].plot(num_epochs, validation_loss, 'x', label = label)
        else:
            ax[0].plot(np.arange(num_epochs), validation_loss, '--o', label = label)
        ax[0].legend()
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss, a.u.')

        ax[1].set_title('Accuracy')
        ax[1].plot(np.arange(num_epochs), train_acc, '--o', label = 'Training')
        if len(validation_loss) == 1:
            ax[1].plot(num_epochs, validation_acc, 'x', label = label)
        else:
            ax[1].plot(np.arange(num_epochs), validation_acc, '--o', label = label)
        ax[1].legend()
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuray')


