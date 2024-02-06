# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
import numpy as np
import os
import glob
import collections

from utils.utils import load_model_pytorch, distributed_is_initialized

from train_cifar10 import ResNet18


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp, optimizers
    USE_APEX = True
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    print("will attempt to run without it")
    USE_APEX = False

#provide intermeiate information
debug_output = False
debug_output = True
device = "cuda" if torch.cuda.is_available() else 'cpu'


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def get_images(net, targets, bs=256, epochs=1000, idx=-1, var_scale=0.00005,
               net_student=None, prefix=None, competitive_scale=0.1, train_writer = None, global_iteration=None,
               use_amp=False,
               optimizer = None, inputs = None, bn_reg_scale = 0.0, random_labels = False, l2_coeff=0.0):
    '''
    Function returns inverted images from the pretrained model, parameters are tight to CIFAR dataset
    args in:
        net: network to be inverted
        bs: batch size
        epochs: total number of iterations to generate inverted images, training longer helps a lot!
        idx: an external flag for printing purposes: only print in the first round, set as -1 to disable
        var_scale: the scaling factor for variance loss regularization. this may vary depending on bs
            larger - more blurred but less noise
        net_student: model to be used for Adaptive DeepInversion
        prefix: defines the path to store images
        competitive_scale: coefficient for Adaptive DeepInversion
        train_writer: tensorboardX object to store intermediate losses
        global_iteration: indexer to be used for tensorboard
        use_amp: boolean to indicate usage of APEX AMP for FP16 calculations - twice faster and less memory on TensorCores
        optimizer: potimizer to be used for model inversion
        inputs: data place holder for optimization, will be reinitialized to noise
        bn_reg_scale: weight for r_feature_regularization
        random_labels: sample labels from random distribution or use columns of the same class
        l2_coeff: coefficient for L2 loss on input
    return:
        A tensor on GPU with shape (bs, 3, 32, 32) for CIFAR
    '''

    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()


    best_cost = 1e6

    # initialize gaussian inputs
    inputs.data = torch.randn((bs, 3, 32, 32), requires_grad=True, device='cuda')
    # if use_amp:
    #     inputs.data = inputs.data.half()

    # set up criteria for optimization
    criterion = nn.CrossEntropyLoss()

    optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer

    # target outputs to generate
    if targets is not None:
        targets = torch.LongTensor([np.random.choice(targets) for _ in range(bs)]).to('cuda')
    else:
        targets = torch.LongTensor([random.randint(0,9) for _ in range(bs)]).to('cuda')

    ## Create hooks for feature statistics catching
    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # setting up the range for jitter
    lim_0, lim_1 = 2, 2

    for epoch in range(epochs):
        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))

        # foward with jit images
        optimizer.zero_grad()
        net.zero_grad()
        outputs = net(inputs_jit)
        
        loss = criterion(outputs, targets)
        loss_target = loss.item()

        # competition loss, Adaptive DeepInvesrion
        if competitive_scale != 0.0 and net_student:

            net_student.zero_grad()
            outputs_student = net_student(inputs_jit)
            T = 3.0

            if 1:
                # jensen shanon divergence:
                # another way to force KL between negative probabilities
                P = F.softmax(outputs_student / T, dim=1)
                Q = F.softmax(outputs / T, dim=1)
                M = 0.5 * (P + Q)

                P = torch.clamp(P, 0.01, 0.99)
                Q = torch.clamp(Q, 0.01, 0.99)
                M = torch.clamp(M, 0.01, 0.99)
                eps = 0.0
                # loss_verifier_cig = 0.5 * kl_loss(F.log_softmax(outputs_verifier / T, dim=1), M) +  0.5 * kl_loss(F.log_softmax(outputs/T, dim=1), M)
                loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                # JS criteria - 0 means full correlation, 1 - means completely different
                loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)
                
                loss = loss + competitive_scale * loss_verifier_cig
#                 print(loss)

        # apply total variation regularization
        diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
        diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
        diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
        diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss = loss + var_scale*loss_var

        # R_feature loss
        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
        loss = loss + bn_reg_scale*loss_distr # best for noise before BN

        # l2 loss
        if 1:
            loss = loss + l2_coeff * torch.norm(inputs_jit, 2)

        if debug_output and epoch % 200==0:
            print(f"It {epoch}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}")
#             vutils.save_image(inputs.data.clone(),
#                               '{}/output_{}.png'.format(prefix, epoch//200),
#                               normalize=True, scale_each=True, nrow=10)

        if best_cost > loss.item():
            best_cost = loss.item()
            best_inputs = inputs.data

        # backward pass
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

    outputs=net(best_inputs)
    _, predicted_teach = outputs.max(1)


    if idx == 0:
        print('Teacher correct out of {}: {}, loss at {}'.format(bs, predicted_teach.eq(targets).sum().item(), criterion(outputs, targets).item()))
        
    name_use = "best_images"
    if prefix is not None:
        name_use = prefix + "/" + name_use
    next_batch = len(glob.glob("./%s/*.png" % name_use)) // 1

    vutils.save_image(best_inputs[:20].clone(),
                      '{}/output_{}.png'.format(name_use, next_batch),
                      normalize=True, scale_each = True, nrow=10)

    if net_student:
        net_student.train()

    return best_inputs, targets


def test():
    print('==> Teacher validation')
    net_teacher.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_teacher(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
def save_images_contiguous(images, targets, prefix, filename_suffix=None):
    
    if filename_suffix:
        torch.save(images.cpu(), prefix + f"/best_inputs_{filename_suffix}.pt")
        torch.save(targets.cpu(), prefix + f"/targets_{filename_suffix}.pt")
        
    else:
        torch.save(images.cpu(), prefix + '/best_inputs.pt')
        torch.save(targets.cpu(), prefix + '/targets.pt')
        

def save_images(images, targets, prefix):
    # method to store generated images locally
    
    for m in targets:
        name = prefix + '/{0:d}'.format(m)
        if not os.path.exists(name):
            os.makedirs(name)
        
    # method to store generated images locally
    local_rank = torch.cuda.current_device()
    for id in range(images.shape[0]):    
        class_id = targets[id].item()
        place_to_store = '{}/img_id{:03d}_2.jpg'.format(prefix, class_id, id)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)
        
def denormalize(image_tensor, in_channels=3, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 DeepInversion')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--iters_mi', default=4000, type=int, help='number of iterations for model inversion')
    parser.add_argument('--cig_scale', default=0, type=float, help='competition score')
    parser.add_argument('--di_lr', default=0.01, type=float, help='lr for deep inversion')
    parser.add_argument('--di_var_scale', default=0.0, type=float, help='TV L2 regularization coefficient')
    parser.add_argument('--di_l2_scale', default=0.0, type=float, help='L2 regularization coefficient')
    parser.add_argument('--r_feature_weight', default=10, type=float, help='weight for BN regularization statistic')
    parser.add_argument('--amp', action='store_true', help='use APEX AMP O1 acceleration')
    parser.add_argument('--exp_descr', default="try1", type=str, help='name to be added to experiment name')
    parser.add_argument('--teacher_weights', default="/project/model_checkpoints/resnet18/resnet18_cifar10_classifer.ckpt", type=str, help='path to load weights of the model')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net_student = ResNet18(2).to(device)
    net_teacher = ResNet18(2).to(device)

    criterion = nn.CrossEntropyLoss()

    # place holder for inputs
    data_type = torch.half if args.amp else torch.float
    inputs = torch.randn((args.bs, 3, 32, 32), requires_grad=True, device='cuda', dtype=data_type)

    optimizer_di = optim.Adam([inputs], lr=args.di_lr)


    if args.amp:
        opt_level = "O1"
        loss_scale = 'dynamic'

        [net_student, net_teacher], optimizer_di = amp.initialize(
            [net_student, net_teacher], optimizer_di,
            opt_level=opt_level,
            loss_scale=loss_scale)

    checkpoint = torch.load(args.teacher_weights)
    net_teacher.load_state_dict(checkpoint)
    net_teacher.eval() #important, otherwise generated images will be non natural
    if args.amp:
        # need to do this trick for FP16 support of batchnorms
        net_teacher.train()
        for module in net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval().half()

    cudnn.benchmark = True


    batch_idx = 0
    prefix = "runs/data_generation/"+args.exp_descr+"/"

    for create_folder in [prefix, prefix+"/best_images/", prefix+"/final_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)

 


    train_writer = None  # tensorboard writter
    global_iteration = 0

    print("Starting model inversion")

    inputs, targets = get_images(net=net_teacher, targets=[0, 1], bs=args.bs, epochs=args.iters_mi, idx=batch_idx,
                        net_student=net_student, prefix=prefix, competitive_scale=args.cig_scale,
                        train_writer=train_writer, global_iteration=global_iteration, use_amp=args.amp,
                        optimizer=optimizer_di, inputs=inputs, bn_reg_scale=args.r_feature_weight, random_labels=True, l2_coeff=args.di_l2_scale)
    
    save_images_contiguous(inputs, targets, prefix+ "/" +"final_images")
    inputs = denormalize(inputs)
    save_images(inputs, targets, prefix+"/final_images/")
    
