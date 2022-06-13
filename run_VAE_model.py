import sys
print(sys.executable)
print(sys.version)
print(sys.version_info)
import socket
print(socket.gethostname())
import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
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
from torchvision.utils import save_image
from sklearn.metrics import r2_score 
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import VAE_functions as func
import argparse
import datetime
# from google.colab import drive

begin_time = datetime.datetime.now()

#0. Read the arguments

parser = argparse.ArgumentParser()

parser.add_argument("-b","--batch_size", help="batch_size", type=int, default = 5)
parser.add_argument("-e", '--num_epochs', help="num_epochs", type=int, default = 20)
parser.add_argument("--beta", help="beta", type=float, default = 1.)
parser.add_argument("-j","--jobID", help="slurm job ID", type=str, default = '')
parser.add_argument("-r","--replicate_n", help="replicate_index", type=str, default = '')
parser.add_argument("--n_lr", help="number of lrs", type=int, default = 1)
parser.add_argument("--min_lr", help="min value lr", type=float, default = 1e-3)
parser.add_argument("--max_lr", help="max value lr", type=float, default = 1e-3)
parser.add_argument("--n_beta", help="number of lrs", type=int, default = 1)
parser.add_argument("--min_beta", help="min value beta", type=float, default = 1)
parser.add_argument("--max_beta", help="max value beta", type=float, default = 1e-2)

args = parser.parse_args()

batch_size = args.batch_size
num_epochs = args.num_epochs
min_lr = args.min_lr
max_lr = args.max_lr
jobID = args.jobID
replicate_n = args.replicate_n
n_lr = args.n_lr
n_beta = args.n_beta
min_beta = args.min_beta
max_beta = args.max_beta

lrs = 10**(np.linspace(np.log10(min_lr),np.log10(max_lr),n_lr))
betas = 10**(np.linspace(np.log10(min_beta),np.log10(max_beta),n_beta))
# beta = 1

#1. Load the dataset 
out_path = '/nfs/scistore08/kondrgrp/emaksimo/GTZAN/models/'
file_name = 'np_db_spec_10genres_99songs.npy'
audio_labels, spectrograms, genres = func.load_spectrograms(file_name)

train_dataset = func.SpectogramDataset(audio_labels, spectrograms, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
valid_dataset = func.SpectogramDataset(audio_labels, spectrograms, valid=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataset = func.SpectogramDataset(audio_labels, spectrograms,)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# train_dataset = TensorDataset(audio_labels, spectograms, train=True, min_max_scaling=True, standardize=False)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
# valid_dataset = TensorDataset(audio_labels, spectograms, valid=True, min_max_scaling=True, standardize=False)
# valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# test_dataset = TensorDataset(audio_labels, spectograms, min_max_scaling=True, standardize=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = func.Set_Device()

#2. Train the model

total_outputs = []

for beta in betas:
    for learning_rate in lrs:

        simple_model = func.VAE_simple(imgChannels=1, batch_size = batch_size).to(device)
        _ = func.print_n_parameters(simple_model)

        (train_loss, train_r2, 
        validation_loss, validation_r2, 
        train_class_loss, validation_class_loss, 
        train_acc, validation_acc) = func.train(simple_model, 
                                               device, 
                                               train_dataloader, 
                                               valid_dataloader, 
                                               num_epochs, 
                                               learning_rate,
                                               beta)

        output = {
            'train_loss':train_loss, 
            'train_r2':train_r2, 
            'validation_loss':validation_loss, 
            'validation_r2':validation_r2, 
            'train_class_loss':train_class_loss, 
            'validation_class_loss':validation_class_loss, 
            'train_acc':train_acc, 
            'validation_acc':validation_acc,
            'batch_size':batch_size,
            'num_epochs':num_epochs,
            'learning_rate':learning_rate,
            'beta':beta,
        }

        total_outputs.append(output)
        model_path = '{}model_{}_lr{:.5f}_b{:.5f}_{}.pt'.format(out_path, jobID, learning_rate, beta, replicate_n)
        print('Saving model in ', model_path)
        torch.save(simple_model, model_path)
        print('Validation accuracy', validation_acc.item(), '%')
        del simple_model
    #     print('Model ', simple_model)
    

#3. Save the output and the model

output_path = '{}output_{}_{}_{}_{}.pkl'.format(out_path, jobID, n_lr, n_beta, replicate_n)
print('Saving output in ', output_path)
torch.save(total_outputs, output_path)

print('Time running: ', datetime.datetime.now() - begin_time)