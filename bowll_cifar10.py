from rtpt import RTPT
import rtpt
rtpt = RTPT(name_initials='XXXX', experiment_name='experiments', max_iterations=10)
rtpt.start()

from random import shuffle
from copy import deepcopy
import argparse
import statistics
import scipy.stats as st
import random

import digit_five.utils as digit5
import cifar10_inversion as deepInv
import bowll.utils as ut
import bowll.active_query as aq
import bowll.open_space_ood as op
from bowll.utils import RingBuffer
from randn_split_dataset import RandSplitCIFAR10, SplitCIFAR10WithNoiseImagenet
from train_cifar10 import ResNet18

import torch
import torch.nn as nn
from torch.nn import init

import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset, RandomSampler, Subset

from collections import deque
import argparse
import math
import os
import gc
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

GPU_MEMPRY_CAP = 2000


def perform_deepInversion(exp_descr, model, targets, bs=5000, iters_mi=4000, cig_scale=0.0, di_lr=0.01, di_var_scale=0.0, di_l2_scale=0.0, r_feature_weight=1, amp=False):

    net_teacher = model
    net_student = deepcopy(model)

    net_student = net_student.to(device)
    net_teacher = net_teacher.to(device)
    

    criterion = nn.CrossEntropyLoss()

    # place holder for inputs
    data_type = torch.half if amp else torch.float
    inputs = torch.randn((bs, 3, 32, 32), requires_grad=True, device='cuda', dtype=data_type)

    optimizer_di = torch.optim.Adam([inputs], lr=di_lr)


    if amp:
        opt_level = "O1"
        loss_scale = 'dynamic'

        [net_student, net_teacher], optimizer_di = amp.initialize(
            [net_student, net_teacher], optimizer_di,
            opt_level=opt_level,
            loss_scale=loss_scale)

    net_teacher.eval() #important, otherwise generated images will be non natural
    if amp:
        # need to do this trick for FP16 support of batchnorms
        net_teacher.train()
        for module in net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval().half()


    batch_idx = 0
    prefix = "/project/runs/data_generation/"+exp_descr

    for create_folder in [prefix, prefix+"/best_images/", prefix+"/final_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)

       
    train_writer = None  # tensorboard writter
    global_iteration = 0

    print("Starting model inversion")

    inputs, targets= deepInv.get_images(net=net_teacher, targets=targets, bs=bs, epochs=iters_mi, idx=batch_idx,
                        net_student=net_student, prefix=prefix, competitive_scale=cig_scale,
                        train_writer=train_writer, global_iteration=global_iteration, use_amp=False,
                        optimizer=optimizer_di, inputs=inputs, bn_reg_scale=r_feature_weight, random_labels=False, l2_coeff=di_l2_scale)
    
    deepInv.save_images_contiguous(inputs, targets, prefix+ "/" +"final_images")
    inputs = deepInv.denormalize(inputs)
    deepInv.save_images(inputs, targets, prefix+ "/" +"final_images")
    
    
    return  prefix+ "/" +"final_images"


def grow_classifier(device, classifier, class_increment):
    new_in_features = classifier[-1].in_features
    new_out_features = classifier[-1].out_features + class_increment
    bias_flag = False

    tmp_weights = classifier[-1].weight.data.clone()
    if not isinstance(classifier[-1].bias, type(None)):
        tmp_bias = classifier[-1].bias.data.clone()
        bias_flag = True

    classifier[-1] = nn.Linear(new_in_features, new_out_features, bias=bias_flag).to(device)

    # copy back the temporarily saved parameters for the slice of previously trained classes.
    classifier[-1].weight.data[0:-class_increment, :] = tmp_weights
    if not isinstance(classifier[-1].bias, type(None)):
        classifier[-1].bias.data[0:-class_increment] = tmp_bias


def train(model, data_loader, aux_data_loader, loss_func, optimizer, num_epochs, device):

    acc = []
    _loss = 0
    _aux_loss = 0
    
    model.train(True)
    
    for ep in range(num_epochs):
        
        for batch_idx, ((x, y), (aux_x, aux_y)) in enumerate(zip(data_loader, aux_data_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)   
            loss = F.cross_entropy(output, y) 
            accuracy = ut.get_accuracy(output, y)
            
            aux_x, aux_y = aux_x.to(device), aux_y.to(device)
            n_output = model(aux_x)
    
            aux_loss = F.cross_entropy(n_output, aux_y) 
            
            total_loss = loss + aux_loss
            _loss += loss.item()
            _aux_loss += aux_loss.item()
            total_loss.backward()
            optimizer.step()
            
        acc.append(accuracy)
        
    avg_acc = np.mean(acc)
    print('overall train accuracy : {:.02f} %'.format(avg_acc * 100))
    return avg_acc,  _loss/num_epochs,  _aux_loss/num_epochs
       
    
def test(model, data_loader, device, timestep, writer=None):
    
    acc = []
    model.eval()

    for batch_idx, (x, target) in enumerate(data_loader):

        x, target = x.to(device), target.to(device)
        output = model(x)

        accuracy = ut.get_accuracy(output, target)
        acc.append(accuracy)
        if writer is not None:
            writer.add_scalar(f"test/timestep-{timestep}/accuracy", accuracy)
        
    avg_acc = np.mean(acc)
    
    print('overall test accuracy : {:.02f} %'.format(avg_acc * 100))
    return avg_acc

   

def args_parser():
    parser = argparse.ArgumentParser(description='BOWLL')
    parser.add_argument('-n', '--experiment_name', type=str, required=True)
    parser.add_argument('--training_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--ood_batch_size', type=int, default=8)
    parser.add_argument('--n_timesteps', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--target_classes', type=int, default=2)
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--acquisition_batch_size', type=int, default=256)
    
    parser.add_argument('--main_dir', type=str, default='/project/experiments')
    
    parser.add_argument('--save_buffer', action='store_true', default=True)
    parser.add_argument('--save_model_checkpoints', action='store_true', default=True)
    
    
    ## model arguments
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', help='model architecture: (default: resnet18)')
    parser.add_argument('-w', '--path_to_weights', help='path to model weights trained on source dataset')
    
    return parser.parse_args()

    
class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.Y[idx]
  
   
def load_model_and_weights(path_to_weights, target_classes, device):
    model = ResNet18(target_classes).to(device)
    model.load_state_dict(torch.load(path_to_weights))
     
    return model 


def main(args):
    
    BATCH_train = args.training_batch_size
    BATCH_test = args.test_batch_size
    BATCH_OOD = args.ood_batch_size
    num_epochs = args.n_epochs
    memory_buffer_size = args.buffer_size
    acquisition_batch_size = args.acquisition_batch_size
    
    main_dir = args.main_dir
    experiment_name = args.experiment_name
    experiment_dir = main_dir + '/' + experiment_name
    
    cifar10_split = RandSplitCIFAR10(train_batch_size=BATCH_train, test_batch_size=BATCH_test, pre_trained_classes=[2, 5])
    cifar10_split.update_task(0)
    
    runs = math.ceil(len(cifar10_split.train_loader.dataset)/BATCH_train)
    bowll_metrics = np.zeros((args.n_repeats, args.n_timesteps, args.n_timesteps, runs)) 
    
  
    for n in range(args.n_repeats):

        random.seed(n)
        np.random.seed(n)
        torch.manual_seed(n)
        torch.cuda.manual_seed(n)

        writer = SummaryWriter(log_dir=f"/project/experiments/{experiment_name}", comment=f"{experiment_name}/_{n}")

        data_queue = RingBuffer(buffer_size=memory_buffer_size, path_to_memory=f"/project/experiments/{experiment_name}/run_{n}", filename='cifar10_buffer')

        model = load_model_and_weights(args.path_to_weights, args.target_classes, device)  

        cifar10_split.update_task(0)

        visited_targets = np.unique(cifar10_split.train_loader.dataset.targets)
        old_model = load_model_and_weights(args.path_to_weights, args.target_classes, device)

        active_learning_data = aq.ActiveLearningDataset(cifar10_split.train_loader.dataset)
        active_learning_data.run_acquisition_randn(memory_buffer_size)
        for images, labels in active_learning_data.training_dataset:
            data_queue.queue.append((images, labels))


        for t in range(1, args.n_timesteps):

            cifar10_split.update_task(t)

            mulvar_g_train_layers = ut.bn_layer_to_mvg(model)
            h_mvg = ut.get_hdim_gaussian(mulvar_g_train_layers, device) 

            data_for_new_loop = list(data_queue.queue)
            bootstrap_dataset = TensorDataset(torch.stack([x[0] for x in data_for_new_loop]), torch.stack([x[1] for x in data_for_new_loop]))

            _, tau = op.compute_tau(model, len(mulvar_g_train_layers), 
                                                       bootstrap_dataset, h_mvg, K=50, M=BATCH_OOD, alpha=0.99)

            ind_indices_with_mi, ood_indices_with_mi = op.compute_ood_score(model, len(mulvar_g_train_layers), 
                                                cifar10_split.train_loader.dataset, BATCH_OOD, h_mvg, tau)

            if(len(ind_indices_with_mi) == 0):
                print('no inliers')

                for d, (_, test_loader) in enumerate(cifar10_split.loaders):
                    acc = test(old_model, test_loader, device, d, writer)
                    bowll_metrics[n][0][d][:] = acc
                for d, (_, test_loader) in enumerate(cifar10_split.loaders):
                    acc = test(model, test_loader, device, d, writer)
                    bowll_metrics[n][t][d][:] = acc
                continue
                
            Y = torch.cat(list(map(list, zip(*ind_indices_with_mi.values())))[0])
            X = torch.cat(list(ind_indices_with_mi.keys()))
            dataset_for_query = CustomDataset(X, Y)

            
            inp_x, label_y = [], []
            class_chunk_for_inversion = 8
            per_class_allocation = math.ceil(memory_buffer_size/len(visited_targets))
            
            for cl in np.array_split(visited_targets, math.ceil(len(visited_targets)/class_chunk_for_inversion)):

                di_batch_size = len(cl)*per_class_allocation
                if di_batch_size > GPU_MEMPRY_CAP:
                    di_batch_size = GPU_MEMPRY_CAP
                
                aux_dataset_path = perform_deepInversion(experiment_name + '_' + str(t), deepcopy(model), cl, bs=di_batch_size)  

                inp_x += torch.load(f"{aux_dataset_path}/best_inputs.pt")
                label_y += torch.load(f"{aux_dataset_path}/targets.pt")

            inp_x, label_y = torch.stack(inp_x), torch.stack(label_y) 
                
            if len(inp_x) < memory_buffer_size:
                inp_x = inp_x.repeat_interleave(memory_buffer_size, dim=0)
                label_y = label_y.repeat_interleave(memory_buffer_size, dim=0)
            aux_dataset = TensorDataset(inp_x, label_y)        

            mask = np.isin(torch.unique(Y).numpy(), visited_targets, assume_unique=True, invert=True)

            if mask.any():
                class_increment = torch.unique(Y).numpy()[mask]
                visited_targets = np.concatenate((class_increment, visited_targets))
                print(visited_targets)
                grow_classifier(device, model.classifer, len(class_increment))

            loss_func = torch.nn.CrossEntropyLoss().to(device) 
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

            active_learning_data = aq.ActiveLearningDataset(dataset_for_query)

            for d, (_, test_loader) in enumerate(cifar10_split.loaders):
                acc = test(old_model, test_loader, device, d, writer)
                bowll_metrics[n][0][d][:] = acc


            n_iters = 0
            per_class_allocation = math.ceil(memory_buffer_size/len(visited_targets))
            
            while len(active_learning_data.pool_dataset) > 0:            

                active_learning_data.run_acquisition(model, acquisition_batch_size, BATCH_train, True, device)
            
                li = []
                for images, labels in active_learning_data.training_dataset:
                    li.append((images, labels))

                n_candidates = data_queue.add_to_buffer(li, model)

                data_for_new_loop = list(data_queue.queue)

                new_timestep_dataset = TensorDataset(torch.stack([x[0] for x in data_for_new_loop]), torch.stack([x[1] for x in data_for_new_loop]))

                aux_data_loader = DataLoader(aux_dataset, batch_size=BATCH_train, shuffle=True)
                cl_data_loader = DataLoader(new_timestep_dataset, batch_size=BATCH_train, shuffle=True)
                avg_acc , _loss, _aux_loss = train(model, cl_data_loader, aux_data_loader, loss_func, optimizer, num_epochs, device)

                if writer:  
                    writer.add_scalar(f"timestep-{t}/train/loss", _loss+_aux_loss, n_iters)
                    writer.add_scalar(f"timestep-{t}/train/aux_loss", _aux_loss, n_iters)
                    writer.add_scalar(f"timestep-{t}/train/clean_loss", _loss, n_iters)


                for d, (_, test_loader) in enumerate(cifar10_split.loaders):
                    acc = test(model, test_loader, device, d, writer)
                    bowll_metrics[n][t][d][n_iters] = acc
                    
                data_queue.saveto_memory(path=None, postfix=f"{t}_{n_iters}")

                n_iters += 1
     
        np.save(f"{main_dir}/{experiment_name}/bowll_metrics_{n}.npy", bowll_metrics)

     

if __name__ == '__main__':
    
    args = args_parser()
    main(args)
    