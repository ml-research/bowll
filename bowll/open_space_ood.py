import typing
import gc
import numpy as np
import numpy.random as npr

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torch.utils.data import DataLoader, Subset, Dataset


device = "cuda" if torch.cuda.is_available() else 'cpu'
s = 1
np.random.seed(s)
torch.manual_seed(s)
torch.cuda.manual_seed(s)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class OpenSpaceOODHook():
    '''
    Forward hook to capture feature statistics: mean and variance
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        output_var, output_mean = torch.var_mean(output.clone().detach(), dim=(0, 2, 3))
        self.bn_var_mean = (torch.mean(torch.nan_to_num(output_var)), torch.mean(output_mean))

    def close(self):
        self.hook.remove()


def compute_eta(net, X_dataloader):
    net.eval()

    bn_hooks = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            bn_hooks.append(OpenSpaceOODHook(layer))
            
    score_list = []
    with torch.no_grad():
        for idx, (inp, target) in enumerate(X_dataloader):
            _ = net(inp.to(device))
            
            x_var = []

            for el in bn_hooks:
                x_var.append(el.bn_var_mean[0])

            x_var = torch.tensor(x_var)
            
            var = torch.mean(x_var)
            
            score = var - torch.log(var) 
            score_list.append(score)
        
    for hook in bn_hooks:
        hook.close()
        
    t_nlls = torch.stack(score_list)
    return t_nlls


def compute_tau(model, X_val, K, M, alpha=0.99):
    N = len(X_val)
    taus = []

    for k in range(K):
        sample_inds = np.random.choice(N, size=M, replace=False)
        val_dataloader = DataLoader(Subset(X_val, sample_inds), batch_size=M)
        
        eta = compute_eta(model, val_dataloader)

        taus.append(eta)

    tau = torch.quantile(torch.Tensor(taus), alpha)
    return taus, tau


def compute_ood_score(net, dataset_to_evalaute, batch_size, tau):
    
    ood_indices_with_mi = dict()
    ind_indices_with_mi = dict()

    net.eval()

    bn_hooks = []
    
    for name, layer in net.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            bn_hooks.append(OpenSpaceOODHook(layer))

    ood_dataloader = DataLoader(dataset_to_evalaute, batch_size=batch_size)
    with torch.no_grad():
        for idx, (inp, target) in enumerate(ood_dataloader):
            _ = net(inp.to(device))
            x_mean = []
            x_var = []

            for el in bn_hooks:
                x_mean.append(el.bn_var_mean[1])
                x_var.append(el.bn_var_mean[0])

            x_var = torch.tensor(x_var)
            x_mean = torch.Tensor(x_mean)

            var = torch.mean(x_var)
            ood_score = var - torch.log(var)

            if ood_score < tau:
                ind_indices_with_mi[inp] = (target, ood_score)

            else:
                ood_indices_with_mi[inp] = (target, ood_score)
                
            gc.collect()
            torch.cuda.empty_cache()
                
    for hook in bn_hooks:
        hook.close()

    return ind_indices_with_mi, ood_indices_with_mi
