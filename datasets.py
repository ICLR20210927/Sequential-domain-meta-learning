import numpy as np
import torch
from torchvision import transforms
from collections import OrderedDict
import os
from torchmeta.datasets import Omniglot, MiniImagenet, CIFARFS
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from CUB import *
from AIRCRAFTloader import *


def dataset(args, datanames):
    #MiniImagenet   
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shot,
                                      num_test_per_class=args.num_query)
    transform = Compose([Resize(84), ToTensor()])
    MiniImagenet_train_dataset = MiniImagenet(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=dataset_transform,
                                      download=True)

    Imagenet_train_loader = BatchMetaDataLoader(MiniImagenet_train_dataset, batch_size=args.MiniImagenet_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    MiniImagenet_val_dataset = MiniImagenet(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=dataset_transform)

    Imagenet_valid_loader = BatchMetaDataLoader(MiniImagenet_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    MiniImagenet_test_dataset = MiniImagenet(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

    Imagenet_test_loader = BatchMetaDataLoader(MiniImagenet_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)


    #CIFARFS
    transform = Compose([Resize(84), ToTensor()])
    CIFARFS_train_dataset = CIFARFS(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=dataset_transform,
                                      download=True)

    CIFARFS_train_loader = BatchMetaDataLoader(CIFARFS_train_dataset, batch_size=args.CIFARFS_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    CIFARFS_val_dataset = CIFARFS(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=dataset_transform)

    CIFARFS_valid_loader = BatchMetaDataLoader(CIFARFS_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    CIFARFS_test_dataset = CIFARFS(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)
    CIFARFS_test_loader = BatchMetaDataLoader(CIFARFS_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)


    #Omniglot
    class_augmentations = [Rotation([90, 180, 270])]
    transform = Compose([Resize(84), ToTensor()])
    Omniglot_train_dataset = Omniglot(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)

    Omniglot_train_loader = BatchMetaDataLoader(Omniglot_train_dataset, batch_size=args.Omniglot_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Omniglot_val_dataset = Omniglot(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)

    Omniglot_valid_loader = BatchMetaDataLoader(Omniglot_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Omniglot_test_dataset = Omniglot(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)
    Omniglot_test_loader = BatchMetaDataLoader(Omniglot_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)


    #CUB dataset
    transform = None
    CUB_train_dataset = CUBdata(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=dataset_transform,
                                      download=False)

    CUB_train_loader = BatchMetaDataLoader(CUB_train_dataset, batch_size=args.CUB_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    CUB_val_dataset = CUBdata(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=dataset_transform)

    CUB_valid_loader = BatchMetaDataLoader(CUB_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    CUB_test_dataset = CUBdata(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)
    CUB_test_loader = BatchMetaDataLoader(CUB_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)
    

    #Aircraftdata
    transform = None
    Aircraft_train_dataset = Aircraftdata(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=dataset_transform,
                                      download=False)


    Aircraft_train_loader = BatchMetaDataLoader(Aircraft_train_dataset, batch_size=args.Aircraft_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Aircraft_val_dataset = Aircraftdata(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=dataset_transform)

    Aircraft_valid_loader = BatchMetaDataLoader(Aircraft_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Aircraft_test_dataset = Aircraftdata(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)
    Aircraft_test_loader = BatchMetaDataLoader(Aircraft_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    
    train_loader_list = []
    valid_loader_list = []
    test_loader_list = []
    for name in  datanames:
        if name == 'MiniImagenet':
            train_loader_list.append({name: Imagenet_train_loader})
            valid_loader_list.append({name: Imagenet_valid_loader})
            test_loader_list.append({name:  Imagenet_test_loader})
        if name == 'CIFARFS':
            train_loader_list.append({name:CIFARFS_train_loader})
            valid_loader_list.append({name:CIFARFS_valid_loader})
            test_loader_list.append({name:CIFARFS_test_loader})
        if name == 'CUB':
            train_loader_list.append({name:CUB_train_loader})
            valid_loader_list.append({name:CUB_valid_loader})
            test_loader_list.append({name:CUB_test_loader})
        if name == 'Aircraft':
            train_loader_list.append({name:Aircraft_train_loader})
            valid_loader_list.append({name:Aircraft_valid_loader})
            test_loader_list.append({name:Aircraft_test_loader})
        if name == 'Omniglot':
            train_loader_list.append({name:Omniglot_train_loader})
            valid_loader_list.append({name:Omniglot_valid_loader})
            test_loader_list.append({name:Omniglot_test_loader})

    return  train_loader_list, valid_loader_list, test_loader_list 


