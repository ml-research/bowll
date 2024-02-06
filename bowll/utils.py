import os
from collections import deque
from random import random, shuffle
from collections import OrderedDict
from copy import deepcopy

import bowll.active_query as aq

import torch
import torch.nn as nn
import torch.distributions as D
from torch.utils.data import Dataset, TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


device = "cuda" if torch.cuda.is_available() else 'cpu'
        


def get_accuracy(output, targets):
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()

def bn_layer_to_mvg(model: nn.Module):
    mvgs = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            means = layer.running_mean.detach().clone()
            covs = torch.diag(layer.running_var.detach().clone())
            mvg_layer = D.MultivariateNormal(means, covs)

            mvgs[name] = mvg_layer

    return mvgs

def get_hdim_gaussian(bn_layers, device):
    h_mean = list()
    h_var = list()

    for ly, mvg in bn_layers.items():
        for i in range(mvg.mean.shape[0]):
            h_mean.append(mvg.mean[i])
            h_var.append(mvg.variance[i])

    covs = torch.diag(torch.stack(h_var)).to(device)
    h_mvg = D.MultivariateNormal(torch.stack(h_mean).to(device), covs)

    return h_mvg

    
    
def plot_ood_distributions(sigma, ood_sigma, epsilon, xlabel, sigma_label, ood_sigma_label, file_name=None):

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    sns.kdeplot(sigma, fill=True, label=sigma_label)
    sns.kdeplot(ood_sigma, fill=True, label=ood_sigma_label)

    ax.set(xlabel=xlabel)

    plt.axvline(epsilon, label='epsilon')
    plt.legend()
    
    if file_name is not None:
        plt.savefig(file_name, dpi=fig.dpi)
        


class RingBuffer:
  
    def __init__(self, path_to_memory=None, filename=None, buffer_size=5000):
        self.buffer_size = buffer_size
        self.queue = deque(maxlen=buffer_size)
        
        if path_to_memory is not None:
            self.path_to_data_checkpoints = path_to_memory
            self.filename = filename
            
            if not os.path.exists(f"{self.path_to_data_checkpoints}/inp"):
                os.makedirs(f"{self.path_to_data_checkpoints}/inp")
                os.makedirs(f"{self.path_to_data_checkpoints}/label")
        
    def load_buffer(self):
        self.z_inp = torch.load(f"{self.path_to_data_checkpoints}/inp/{self.filename}.pt")
        self.z_label = torch.load(f"{self.path_to_data_checkpoints}/label/{self.filename}.pt")
        
        self.fill_queue()
        
    def fill_queue(self):
        data = list(zip(self.z_inp, self.z_label))
        list(map(self.queue.append, data))
        shuffle(self.queue)
        
    def empty_buffer(self):
        del self.z_inp
        del self.z_label
        torch.cuda.empty_cache()
        
        os.remove(f"{path_to_data_checkpoints}/inp/{filename}.pt")
        os.remove(f"{path_to_data_checkpoints}/label/{filename}.pt")
        
    def saveto_memory(self, path, postfix, empty_buffer=False):
        if empty_buffer:
            self.empty_buffer()
        
        save_tensor_inp = []
        save_tensor_label = []
        for x, y in list(self.queue):
            save_tensor_inp.append(x)  
            save_tensor_label.append(y) 
            
        if path is None:
            path = self.path_to_data_checkpoints
            
        torch.save(torch.stack(save_tensor_inp), f"{path}/inp/{self.filename}_{postfix}.pt")
        torch.save(torch.stack(save_tensor_label), f"{path}/label/{self.filename}_{postfix}.pt")
                

    def add_list_to_buffer(self, batch):
        while batch:
            self.queue.append(batch.pop())
            
   
    def add_to_buffer(self, data, model, batch_size=256):
        inp_x = []
        inp_y = []
        
        idx = 0
        for x, y in list(self.queue):
            inp_x.append(x)
            inp_y.append(y)

            idx = idx + 1
        
        indices = []
        for image, label in data:

            inp_x.append(image)
            inp_y.append(label)

            indices.append(idx) 
            idx = idx + 1   

                    
        inp_y = torch.stack(inp_y)
        dataset_for_query = TensorDataset(torch.stack(inp_x), inp_y)
        
        active_learning_data = aq.ActiveLearningDataset(dataset_for_query)
        acquisition_batch_size = self.buffer_size
        
        candidate_indices  = active_learning_data.run_acquisition(model, acquisition_batch_size, batch_size, False, device)

        new_candidates = list(set(candidate_indices) & set(indices))
        if new_candidates:
            print(len(new_candidates))
    

        images, labels = deepcopy(active_learning_data.training_dataset[:])
        li = list(zip(images, labels))
        
        self.add_list_to_buffer(li)
        self.add_list_to_buffer(data)

        return len(new_candidates)
                             
        
    def size(self):
        # return the size of the queue
        return self.buffer_size
    
    def get_alldata(self):
        return list(self.queue)
    
    def get_dataloader(self, batch_size=256):
        inp_x = []
        inp_y = []
        
        idx = 0
        for x, y in list(self.queue):
            inp_x.append(x)
            inp_y.append(y)

        dataset_for_query = TensorDataset(torch.stack(inp_x), torch.stack(inp_y))
        dataloader = DataLoader(dataset_for_query, batch_size=batch_size, shuffle=False)

        return dataloader
        
        
