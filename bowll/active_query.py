from typing import List
import enum
import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as D
from torch.utils.data import DataLoader, Dataset, Subset

import bowll.utils as ut
from  bowll.open_space_ood import OpenSpaceOODHook


def get_top_k_scorers(scores_N, batch_size, uncertainty=True):
    N = len(scores_N)
    batch_size = min(batch_size, N)
    candidate_scores, candidate_indices = torch.topk(scores_N, batch_size, largest=uncertainty)
    return candidate_scores.tolist(), candidate_indices.tolist()


def gaussian_entropy(sample):

    half_log_det = sample.log().sum(-1)
    H = 0.5 * sample.shape[-1] * (1.0 + math.log(2 * math.pi)) + half_log_det
    
    return H

class ActiveLearningDataset:
    
    dataset: Dataset
    training_dataset: Dataset
    pool_dataset: Dataset
    training_mask: np.ndarray
    pool_mask: np.ndarray

    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset
        self.training_mask = np.full((len(dataset),), False)
        self.pool_mask = np.full((len(dataset),), True)

        self.training_dataset = Subset(self.dataset, None)
        self.pool_dataset = Subset(self.dataset, None)

        self._update_indices()

    def _update_indices(self):
        self.training_dataset.indices = np.nonzero(self.training_mask)[0]
        self.pool_dataset.indices = np.nonzero(self.pool_mask)[0]

    def get_dataset_indices(self, pool_indices: List[int]) -> List[int]:
        """Transform indices (in `pool_dataset`) to indices in the original `dataset`."""
        indices = self.pool_dataset.indices[pool_indices]
        return indices

    def acquire(self, pool_indices):
        """
        Acquire elements from the pool dataset into the training dataset.
        Add them to training dataset & remove them from the pool dataset.
        """
        indices = self.get_dataset_indices(pool_indices)
        
        self.training_mask = np.full((len(self.dataset),), False)
  
        self.training_mask[indices] = True
        self.pool_mask[indices] = False
        
        self._update_indices()

    def remove_from_pool(self, pool_indices):
        """
        Remove from the pool dataset.
        """
        indices = self.get_dataset_indices(pool_indices)

        self.pool_mask[indices] = False
        self._update_indices()

    def get_random_pool_indices(self, size) -> torch.LongTensor:
        assert 0 <= size <= len(self.pool_dataset)
        pool_indices = torch.randperm(len(self.pool_dataset))[:size]
        return pool_indices

    def extract_dataset_from_pool(self, size) -> Dataset:
        """
        Extract a dataset randomly from the pool dataset and make those indices unavailable.
        Useful for extracting a validation set.
        """
        return self.extract_dataset_from_pool_from_indices(self.get_random_pool_indices(size))

    def extract_dataset_from_pool_from_indices(self, pool_indices) -> Dataset:
        """
        Extract a dataset from the pool dataset and make those indices unavailable.
        Useful for extracting a validation set.
        """
        dataset_indices = self.get_dataset_indices(pool_indices)

        self.remove_from_pool(pool_indices)
        return Subset(self.dataset, dataset_indices)
    
    def get_random_pool_indices(self, size) -> torch.LongTensor:
        if size > len(self.pool_dataset):
            size = len(self.pool_dataset)
            
        pool_indices = torch.randperm(len(self.pool_dataset))[:size]
        return pool_indices
    
    def run_acquisition_randn(self, pool_selection_size=256):
        self.acquire(self.get_random_pool_indices(pool_selection_size))
        
        
    def run_acquisition(self, net: nn.Module, acquisition_size=256, train_batch_size=256, cosine_sim=True, device=None):
        net.eval()
        
        dataloader = DataLoader(self.pool_dataset, batch_size=train_batch_size, shuffle=False)

        bn_hooks = []
        for name, layer in net.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                bn_hooks.append(OpenSpaceOODHook(layer))  
  
        query_scores = []

        with torch.no_grad():
            means = []
            entropys = []
            for idx, (inp, target) in enumerate(dataloader):
                inp, target = inp.to(device), target.to(device)
                _ = net(inp)

                x_var = []
                for el in bn_hooks:
                    x_var.append(el.bn_var_mean[0])

                x_var = torch.stack(x_var).transpose(0,1)
                entropy = gaussian_entropy(x_var)

                inp_x = inp.reshape((len(inp), -1))
                means.append(inp_x.detach().cpu())
                entropys.append(entropy.detach().cpu())
                
        
        means = torch.cat(means)
        entropys = torch.cat(entropys)
        similarity_measure = cosine_similarity(means)
        similarity_measure = similarity_measure[~torch.eye(similarity_measure.shape[0],dtype=bool)].reshape(similarity_measure.shape[0],-1)
        
        for idx, _ in enumerate(means):
            if cosine_sim:
                score = entropys[idx] * (torch.tensor(similarity_measure[idx][:].mean()))
            else:
                score = entropys[idx] * (torch.tensor(1-similarity_measure[idx][:].mean()))
            query_scores.append(score)


        if query_scores:
            (candidate_scores, candidate_indices,) = get_top_k_scorers(torch.tensor(query_scores), acquisition_size) 
            
            self.acquire(candidate_indices)

        return candidate_indices
    

class ActiveQueryMethod(enum.Enum):
    random = "random"
    query_by_novelty = "novelty_score"
    