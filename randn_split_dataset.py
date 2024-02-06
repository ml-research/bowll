import numpy as np

import copy

import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import ConcatDataset, Dataset


def partition_dataset(dataset, perm, org_labels):
    lperm = perm.tolist()
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label in lperm
    ]

    newdataset.targets = [
        org_labels.index(label)
        for label in newdataset.targets
        if label in lperm
    ]
    return newdataset

def partition_multiple_dataset(dataset_list, perm, org_labels):
    lperm = perm.tolist()
    
    datasets = []
    for d in dataset_list:
        newdataset = copy.copy(d)
        newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label in lperm
    ]

        newdataset.targets = [
        org_labels.index(label)
        for label in newdataset.targets
        if label in lperm
    ] 
        datasets.append(newdataset)
        
    newdataset = ConcatDataset(datasets)
    return newdataset

class RandSplitCIFAR10:
    def __init__(self, target_classes=2, n_timesteps=5, total_cls = 10, pre_trained_classes=[2, 5], train_batch_size=256, test_batch_size=256, data_path='/project/data_cifar10'):
        super(RandSplitCIFAR10, self).__init__()
        num_classes = target_classes
        data_root = data_path
        total_cls = total_cls

        use_cuda = torch.cuda.is_available()

        mean = (0.491, 0.482, 0.447)
        std = (0.247, 0.243, 0.262)
        normalize = transforms.Normalize(
            mean=mean, std=std
        )
       
        perm = np.arange(total_cls)
        perm = np.asarray([x for x in perm if x not in pre_trained_classes])
        
#         remapped_labels = {x:i for i, x in enumerate(remapped_classes)}
        r_target_transform=transforms.Lambda(lambda y: torch.tensor(y))
        
        org_labels = []+pre_trained_classes
        for i in range(n_timesteps-1):
            org_labels += perm[i::n_timesteps-1].tolist()
            
        print(org_labels)
        
        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=False,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            target_transform=r_target_transform,
        )
        
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
            target_transform=r_target_transform,
        )

        

        splits = [
            (
                partition_dataset(train_dataset, perm[i::n_timesteps-1], org_labels),
                partition_dataset(val_dataset, perm[i::n_timesteps-1], org_labels),
            )
            for i in range(n_timesteps-1)
        ]
        
        splits.insert(0, (partition_dataset(train_dataset, np.asarray(pre_trained_classes), org_labels),
                partition_dataset(val_dataset, np.asarray(pre_trained_classes), org_labels)))

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=train_batch_size, shuffle=True
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=test_batch_size, shuffle=True
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]
        
class RandSplitCIFAR100:
    def __init__(self, target_classes=10, n_timesteps=10, total_cls = 100, pre_trained_classes=[0,1,2], train_batch_size=256, test_batch_size=256, data_path='/project/cifar100'):
        
        super(RandSplitCIFAR100, self).__init__()
        num_classes = target_classes
        data_root = data_path
        total_cls = total_cls

        use_cuda = torch.cuda.is_available()

        mean = (0.491, 0.482, 0.447)
        std = (0.247, 0.243, 0.262)
        normalize = transforms.Normalize(
            mean=mean, std=std
        )
       
        perm = np.arange(total_cls)
        perm = np.asarray([x for x in perm if x not in pre_trained_classes])
        
#         remapped_labels = {x:i for i, x in enumerate(remapped_classes)}
        r_target_transform=transforms.Lambda(lambda y: torch.tensor(y))
        
        org_labels = []+pre_trained_classes
        for i in range(n_timesteps-1):
            org_labels += perm[i::n_timesteps-1].tolist()
            
        print(org_labels)
        
        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            target_transform=r_target_transform,
        )
        
        val_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
            target_transform=r_target_transform,
        )

        splits = [
            (
                partition_dataset(train_dataset, perm[i::n_timesteps-1], org_labels),
                partition_dataset(val_dataset, perm[i::n_timesteps-1], org_labels),
            )
            for i in range(n_timesteps-1)
        ]
        
        splits.insert(0, (partition_dataset(train_dataset, np.asarray(pre_trained_classes), org_labels),
                partition_dataset(val_dataset, np.asarray(pre_trained_classes), org_labels)))

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=train_batch_size, shuffle=True
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=test_batch_size, shuffle=True
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]

class CustomTensorDataset(Dataset):
    def __init__(self, X, Y, transforms):
        self.data = X
        self.targets = Y
        self.transforms = transforms

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.transforms(self.data[idx]), torch.tensor(self.targets[idx])
    
class CustomSubsetDataset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels, transforms=None):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels
        self.indices = indices
        self.data = [self.dataset[idx][0] for idx in indices]
        self.transforms = transforms 
        
    def __getitem__(self, idx):
        image = self.data[idx]
        target = torch.tensor(self.targets[idx])
        
        if self.transforms:
            image = self.transforms(image)
        
        return (image, target)

    def __len__(self):
        return len(self.targets)

    
        
class SplitCIFAR10WithNoiseImagenet:
    def __init__(self, target_classes=2, n_timesteps=5, total_cls = 10, pre_trained_classes=[2, 5], train_batch_size=256, test_batch_size=256, data_path='/project/data_cifar10'):
        super(SplitCIFAR10WithNoiseImagenet, self).__init__()
        num_classes = target_classes
        data_root = data_path
        total_cls = total_cls

        use_cuda = torch.cuda.is_available()

        mean = (0.491, 0.482, 0.447)
        std = (0.247, 0.243, 0.262)
        normalize = transforms.Normalize(
            mean=mean, std=std
        )

        # cifar10_Im = torch.from_numpy(np.transpose(np.load('/project/data_cifar10/CIFAR-10-C/impulse_noise.npy'), (0, 3, 1, 2))).to(torch.float)
        # cifar10_Im_label = torch.from_numpy(np.load('/project/data_cifar10/CIFAR-10-C/labels.npy'))
        
        # cifar10_G = torch.from_numpy(np.transpose(np.load('/project/data_cifar10/CIFAR-10-C/gaussian_noise.npy'), (0, 3, 1, 2))).to(torch.float)
        # cifar10_G_label = torch.from_numpy(np.load('/project/data_cifar10/CIFAR-10-C/labels.npy'))

        # cifar10_shotNoise = torch.from_numpy(np.transpose(np.load('/project/data_cifar10/CIFAR-10-C/shot_noise.npy'), (0, 3, 1, 2))).to(torch.float)
        # cifar10_shotNoise_label = torch.from_numpy(np.load('/project/data_cifar10/CIFAR-10-C/labels.npy'))

        # cifar10_corrupted = torch.cat([cifar10_Im, cifar10_G, cifar10_shotNoise])
        # cifar10_corrupted_label = torch.cat([cifar10_Im_label, cifar10_G_label, cifar10_shotNoise_label])
       
        perm = np.arange(total_cls)
        perm = np.asarray([x for x in perm if x not in pre_trained_classes])
        
        transform_to_T = transforms.Compose([transforms.ToTensor()])
        r_target_transform=transforms.Lambda(lambda y: torch.tensor(y))
        
        org_labels = []+pre_trained_classes
        for i in range(n_timesteps-1):
            org_labels += perm[i::n_timesteps-1].tolist()
            
        print(org_labels)
        
        transform_all=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        
        imagenetNw = datasets.ImageNet(root='/imagenet', transform=transform_all, split= 'val', download=False) 
        indices_val = [i for i, label in enumerate(imagenetNw.targets) if label in org_labels]
        indices_targets = [label for i, label in enumerate(imagenetNw.targets) if label in org_labels]
        imagenet_extra = CustomSubsetDataset(imagenetNw, indices_val, indices_targets)
      
        # transform_noisy = transforms.Compose([transforms.Resize((32, 32)), normalize])
        # noisy_dataset = CustomTensorDataset(cifar10_corrupted, cifar10_corrupted_label, transform_noisy)
        
        
        cifar10_train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=False,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            target_transform=r_target_transform,
        )
        # train_dataset = [cifar10_train_dataset, noisy_dataset, imagenet_extra]
        train_dataset = [cifar10_train_dataset, imagenet_extra]
        
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
            target_transform=r_target_transform,
        )
        

        splits = [
            (
                partition_multiple_dataset(train_dataset, perm[i::n_timesteps-1], org_labels),
                partition_dataset(val_dataset, perm[i::n_timesteps-1], org_labels),
            )
            for i in range(n_timesteps-1)
        ]
        
        splits.insert(0, (partition_dataset(cifar10_train_dataset, np.asarray(pre_trained_classes), org_labels),
                partition_dataset(val_dataset, np.asarray(pre_trained_classes), org_labels)))

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=train_batch_size, shuffle=True
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=test_batch_size, shuffle=True
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]
        
        