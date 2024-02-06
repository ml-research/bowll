from rtpt import RTPT
import rtpt
rtpt = RTPT(name_initials='XXXX', experiment_name='experiments', max_iterations=10)
rtpt.start()

from random import shuffle
import random
from copy import deepcopy
import argparse

import digit_five.utils as digit5
import mnist_inv as deepInv
import bowll.utils as ut
import bowll.open_space_ood as op
import bowll.active_query as aq
from bowll.utils import RingBuffer
from alexnet_bn import AlexNetBN

import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset, RandomSampler

from collections import deque
import argparse
import math
import os
import shutil
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams['figure.constrained_layout.use'] = True

device = "cuda" if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def perform_deepInversion(exp_descr, model, bs=5000, iters_mi=3000, cig_scale=0.0, di_lr=0.01, di_var_scale=0.001, di_l2_scale=0.0001, r_feature_weight=0.1, amp=False):

    net_teacher = model
    net_student = deepcopy(net_teacher)
    
    criterion = nn.CrossEntropyLoss()

    # place holder for inputs
    data_type = torch.half if amp else torch.float
    inputs = torch.randn((bs, 1, 28, 28), requires_grad=True, device='cuda', dtype=data_type)

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

    if 1:
        # loading
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
       
    train_writer = None  # tensorboard writter
    global_iteration = 0

    print("Starting model inversion")

    inputs, targets= deepInv.get_images(net=net_teacher, bs=bs, epochs=iters_mi, idx=batch_idx,
                        net_student=net_student, prefix=prefix, competitive_scale=cig_scale,
                        train_writer=train_writer, global_iteration=global_iteration, use_amp=False,
                        optimizer=optimizer_di, inputs=inputs, bn_reg_scale=r_feature_weight, random_labels=True, l2_coeff=di_l2_scale)
    
    inputs = deepInv.denormalize(inputs)
    deepInv.save_images(inputs, targets, prefix+ "/" +"final_images")
    deepInv.save_images_contiguous(inputs, targets, prefix+ "/" +"final_images")
    
    return  prefix+ "/" +"final_images"


    
def train(model, data_loader, aux_data_loader, loss_func, optimizer, num_epochs, device, writer, t, aq_run):

    acc = []
    
    model.train(True)
    
    for ep in range(num_epochs):
        _loss = 0
        for batch_idx, ((x, y), (aux_x, aux_y)) in enumerate(zip(data_loader, aux_data_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)   
            loss = F.cross_entropy(output, y) 
            accuracy = ut.get_accuracy(output, y)
            
            aux_x, aux_y = aux_x.to(device), aux_y.to(device)
            n_output = model(aux_x)
            kl_loss = F.cross_entropy(n_output, aux_y)          
          
            total_loss = loss + kl_loss
            _loss += total_loss.item()
            
            total_loss.backward()
            optimizer.step()
            
        acc.append(accuracy)

        if writer is not None:
            writer.add_scalar(f"timestep-{t}/aq-{aq_run}/train/loss", _loss, ep)
            writer.add_scalar(f"timestep-{t}/aq-{aq_run}/train/accuracy", accuracy, ep)     
        
    avg_acc = np.mean(acc)
    print('overall train accuracy : {:.02f} %'.format(avg_acc * 100))
    

    
def test(model, data_loader, device, writer=None):
    
    acc = []
    model.eval()

    for batch_idx, (x, target) in enumerate(data_loader):

        x, target = x.to(device), target.to(device)
        output = model(x)

        accuracy = ut.get_accuracy(output, target)
        acc.append(accuracy)
        if writer is not None:
            writer.add_scalar("test/accuracy", accuracy)
        
    avg_acc = np.mean(acc)
    print('overall test accuracy : {:.02f} %'.format(avg_acc * 100))
    return avg_acc


def args_parser():
    parser = argparse.ArgumentParser(description='BOWLL')
    parser.add_argument('-n', '--experiment_name', type=str, default='exp_bowll_mnist')
    parser.add_argument('--training_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--ood_batch_size', type=int, default=4)
    parser.add_argument('--n_domains', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--acquisition_batch_size', type=int, default=256)
    
    parser.add_argument('--main_dir', type=str, default='/project/experiments')
    
    parser.add_argument('--save_buffer', action='store_true', default=True)
   
    parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet', help='model architecture: (default: resnet18)')
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
  
   
def load_model_and_weights(path_to_weights, device):
    model = AlexNetBN(1, 10).to(device)
    model.load_state_dict(torch.load(path_to_weights))
     
    return model

def get_domainlearnig_data(BATCH_train, BATCH_test):
    
    transform_rgb_to_grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Resize(28), transforms.Normalize((0.45,), (0.2,))])
    
    test_set_svhn = datasets.SVHN('/project/mnist_svhn/data_svhn/', download=True, split='test', transform=transform_rgb_to_grayscale)
    train_set_svhn = datasets.SVHN('/project/mnist_svhn/data_svhn/', download=True, split='train', transform=transform_rgb_to_grayscale)
    
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])

    train_set_mnist = datasets.MNIST('/project/mnist_svhn/data_mnist/', download=True, train=True, transform=transform_to_tensor, target_transform=transforms.Lambda(lambda y: torch.tensor(y)))
    test_set_mnist = datasets.MNIST('/project/mnist_svhn/data_mnist/', download=True, train=False, transform=transform_to_tensor, target_transform=transforms.Lambda(lambda y: torch.tensor(y)))
    mnist_train_dataloader = torch.utils.data.DataLoader(train_set_mnist, batch_size=BATCH_train, shuffle=True)
    mnist_test_dataloader = torch.utils.data.DataLoader(test_set_mnist, batch_size=BATCH_test, shuffle=True)
    
    digit5_traindataloader, digit5_testdataloder = digit5.get_dataloader(path='/project/digit_five/Digit-Five')
    
    svhn_train_loader = DataLoader(train_set_svhn, batch_size=BATCH_test, shuffle=True)
    svhn_test_loader = DataLoader(test_set_svhn, batch_size=BATCH_test, shuffle=True)
    
    domain_dict = {0:[mnist_train_dataloader, mnist_test_dataloader], 1:[svhn_train_loader, svhn_test_loader], 2:[digit5_traindataloader, digit5_testdataloder]}
    
    return domain_dict


def main(args):
    
    BATCH_train = args.training_batch_size
    BATCH_test = args.test_batch_size
    BATCH_OOD = args.ood_batch_size
    num_epochs = args.n_epochs
    memory_buffer_size = args.buffer_size
    acquisition_batch_size = args.acquisition_batch_size
    
    
    main_dir = args.main_dir
    experiment_name = f"{args.experiment_name}"
    experiment_dir = main_dir + '/' + experiment_name
    writer = SummaryWriter(log_dir=experiment_dir, comment='')
    bowll_metrics = None

    dataloader_dicts = get_domainlearnig_data(BATCH_train, BATCH_test)
    
    n_runs = math.ceil(len(dataloader_dicts[1][0].dataset)/BATCH_train)
    bowll_metrics = np.zeros((args.n_repeats, args.n_domains, args.n_domains, n_runs))
  
    for n in range(args.n_repeats):
        
        random.seed(n)
        np.random.seed(n)
        torch.manual_seed(n)
        torch.cuda.manual_seed(n)

        model = load_model_and_weights(args.path_to_weights, device)
   
        data_queue = RingBuffer(buffer_size=memory_buffer_size, path_to_memory=f"{experiment_dir}/run_{n}", filename='_buffer')
        active_learning_data = aq.ActiveLearningDataset(dataloader_dicts[0][0].dataset)
        active_learning_data.run_acquisition_randn(memory_buffer_size)
        for images, labels in active_learning_data.training_dataset:
            data_queue.queue.append((images, labels))
    
 
        for t in range(1, args.n_domains):

            mulvar_g_train_layers = ut.bn_layer_to_mvg(model)
            h_mvg = ut.get_hdim_gaussian(mulvar_g_train_layers, device)  
            
            aux_dataset_path = perform_deepInversion(experiment_name + '_' + str(t), deepcopy(model), bs=256*10)
           
            aux_x = torch.load(f"{aux_dataset_path}/best_images.pt").cpu()
            aux_y = torch.load(f"{aux_dataset_path}/targets.pt").cpu()
            aux_dataset = TensorDataset(aux_x,aux_y)
            
            data_for_new_loop = list(data_queue.queue)
            bootstrap_data = TensorDataset(torch.stack([x[0] for x in data_for_new_loop]), torch.stack([x[1] for x in data_for_new_loop]))

            _, tau = op.compute_tau(model, bootstrap_data, h_mvg, K=50, M=BATCH_OOD, alpha=0.99)

            ind_indices_with_mi, ood_indices_with_mi = op.compute_ood_score(model, dataloader_dicts[t][0].dataset, BATCH_OOD, tau)

            Y = torch.cat(list(map(list, zip(*ind_indices_with_mi.values())))[0])
            X = torch.cat(list(ind_indices_with_mi.keys()))
           
            dataset_for_query = CustomDataset(X, Y)

            aux_x = aux_x.repeat_interleave(memory_buffer_size, dim=0)
            aux_y = aux_y.repeat_interleave(memory_buffer_size, dim=0)
            aux_dataset = TensorDataset(aux_x,aux_y)

            loss_func = torch.nn.CrossEntropyLoss().to(device)    
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            active_learning_data = aq.ActiveLearningDataset(dataset_for_query)
            
            for d, loaders in dataloader_dicts.items():
                acc = test(model, loaders[-1], device, writer)
                bowll_metrics[n][0][d][:] = acc

            n_aq_runs = 0
            while len(active_learning_data.pool_dataset) > 0:
                
                print(len(active_learning_data.pool_dataset))

                active_learning_data.run_acquisition(model, acquisition_batch_size, BATCH_train, True, device)
                print('finished acquiring...')
                
                data_for_queue = []
                for data in active_learning_data.training_dataset:
                    data_for_queue.append(data)
                    
                n_candidates = data_queue.add_to_buffer(data_for_queue, model)
                
                print('finished queueing...')
                data_for_new_loop = list(data_queue.queue)

                bcbcbc = TensorDataset(torch.stack([x[0] for x in data_for_new_loop]), torch.stack([x[1] for x in data_for_new_loop]))
                
                aux_data_loader = DataLoader(aux_dataset, batch_size=BATCH_train, shuffle=True)
                cl_data_loader = DataLoader(bcbcbc, batch_size=BATCH_train, shuffle=True)

                train(model, cl_data_loader, aux_data_loader, loss_func, optimizer, num_epochs, device, writer, t, n_aq_runs)
                
                for d, loaders in dataloader_dicts.items():
                    acc = test(model, loaders[-1], device, writer)
                    bowll_metrics[n][t][d][n_aq_runs] = acc
                    
                data_queue.saveto_memory(path=None, postfix=f"{t}_{n_aq_runs}")
                n_aq_runs += 1
                    

        np.save(f"{main_dir}/{experiment_name}/bowll_metrics_{n}.npy", bowll_metrics)  


        
   
if __name__ == '__main__':
    
    args = args_parser()
    main(args)
    