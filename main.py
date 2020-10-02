#some codes are borrowed from https://github.com/tristandeleu/pytorch-meta/tree/master/examples/protonet

import os
import torch
from tqdm import tqdm
from model_filter import PrototypicalNetworkhead1
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from utils import get_accuracy
import numpy as np
from datasets import *
import pickle
import random
from MuAdam_filter import *
import time
import copy

datanames = ['MiniImagenet', 'CIFARFS', 'Omniglot', 'Aircraft', 'CUB']

class SequentialMeta(object):
    def __init__(self,model, lr=0.001, args=None):
        self.args = args
        self.model=model
        self.init_lr=lr
        self.hyper_lr = args.hyper_lr
        self.update_lr(domain_id=0, lr=1e-3)
        self.hyper_optim = MuAdam(self.optimizer, self.args.hyper_lr, True, self.args.device, self.args.clip_hyper, self.args.LR, self.args.layer_filters)
        str_save = '_'.join(datanames)
        self.filepath = os.path.join(self.args.output_folder, 'protonet_LRschedule{}'.format(str_save), 'Block{}'.format(self.args.num_block), 'shot{}'.format(self.args.num_shot), 'way{}'.format(self.args.num_way))
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

    def train(self, epoch, dataloader_dict, memory_train = None, domain_id = None):
        self.model.train()
        for dataname, dataloader in dataloader_dict.items():
            with tqdm(dataloader, total=self.args.num_batches) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    self.model.zero_grad()
                    train_inputs, train_targets = batch['train']
                    train_inputs = train_inputs.to(device=self.args.device)
                    train_targets = train_targets.to(device=self.args.device)
                    if train_inputs.size(2) == 1:
                        train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
                    train_embeddings = self.model(train_inputs, domain_id)
                    test_inputs, test_targets = batch['test']
                    test_inputs = test_inputs.to(device=self.args.device)
                    test_targets = test_targets.to(device=self.args.device)
                    if test_inputs.size(2) == 1:
                        test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                    test_embeddings = self.model(test_inputs, domain_id)

                    prototypes = get_prototypes(train_embeddings, train_targets, args.num_way)
                    loss = prototypical_loss(prototypes, test_embeddings, test_targets)
                    loss.backward(retain_graph=True)
                    
                    param_list = []
                    param_names = []
                    for name, v in self.model.named_parameters():
                        if 'domain_out' not in name: 
                            if v.requires_grad:
                                param_list.append(v)
                                param_names.append(name)
                    first_grad = torch.autograd.grad(loss, param_list, create_graph=False, retain_graph=False)

                    val_graddict = {}
                    layer_name = []
                    for gradient, name in zip(first_grad, param_names):
                        split_name = name.split('.')
                        layer = split_name[0]
                        if layer not in self.args.layer_filters:
                            if layer not in layer_name:
                                layer_name.append(layer)
                                val_graddict[layer] = []
                                val_graddict[layer].append(gradient.clone().view(-1))
                            else:
                                val_graddict[layer].append(gradient.clone().view(-1))
                        else:
                            layer_sub = layer+'.'+split_name[1]+'.'+split_name[2]
                            if layer_sub not in layer_name:
                                layer_name.append(layer_sub)
                                val_graddict[layer_sub] = []
                                val_graddict[layer_sub].append(gradient.clone().view(-1))
                            else:
                                val_graddict[layer_sub].append(gradient.clone().view(-1))

                    for key in val_graddict:
                        val_graddict[key] = torch.cat(val_graddict[key])
                    self.optimizer.step()
                    
                    if memory_train:
                        memory_trainnew = copy.deepcopy(memory_train)
                        self.hyper_optim.optimizer = self.optimizer
                        self.hyper_optim.compute_hg(self.model, val_graddict)
                        val_grad = self.rep_grad_new(self.args, memory_trainnew)
                        
                        self.hyper_optim.hyper_step(val_grad)
                        self.model.zero_grad()
                    
                    if batch_idx >= args.num_batches:
                        break


    def rep_grad_new(self, args, memory_train):
        memory_loss =0
        for dataidx, dataloader_dict in enumerate(memory_train):
                for dataname, memory_list in dataloader_dict.items():
                    select = random.choice(memory_list)
                    memory_train_inputs, memory_train_targets = select['train'] 
                    memory_train_inputs = memory_train_inputs.to(device=args.device)
                    memory_train_targets = memory_train_targets.to(device=args.device)
                    if memory_train_inputs.size(2) == 1:
                        memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
                    memory_train_embeddings = self.model(memory_train_inputs, dataidx)

                    memory_test_inputs, memory_test_targets = select['test'] 
                    memory_test_inputs = memory_test_inputs.to(device=args.device)
                    memory_test_targets = memory_test_targets.to(device=args.device)
                    if memory_test_inputs.size(2) == 1:
                        memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)
                   

                indlist = []
                for ind in range(len(memory_train)):
                    if ind != dataidx:
                        indlist.append(ind)

                if indlist:
                    indselect = random.choice(indlist)      
                    dataloader_dict2 = memory_train[indselect] 
                    for dataname, memory_list in dataloader_dict2.items():
                        select2 = random.choice(memory_list)
                        memory_train_inputs2, memory_train_targets2 = select2['train'] 
                        memory_train_inputs2 = memory_train_inputs2.to(device=args.device)
                        memory_train_targets2 = memory_train_targets2.to(device=args.device)
                        if memory_train_inputs2.size(2) == 1:
                            memory_train_inputs2 = memory_train_inputs2.repeat(1, 1, 3, 1, 1)
                        memory_train_embeddings2 = self.model(memory_train_inputs2, dataidx)

                        memory_test_inputs2, memory_test_targets2 = select2['test'] 
                        memory_test_inputs2 = memory_test_inputs2.to(device=args.device)
                        memory_test_targets2 = memory_test_targets2.to(device=args.device)
                        if memory_test_inputs2.size(2) == 1:
                            memory_test_inputs2 = memory_test_inputs2.repeat(1, 1, 3, 1, 1)
                    
                    memory_test_embeddings2 = self.model(memory_test_inputs2, dataidx)
                memory_test_embeddings = self.model(memory_test_inputs, dataidx)
                memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, args.num_way)
                memory_loss += prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)

                if indlist:
                    sellist = range(len(memory_train_embeddings))
                    i = random.choice(sellist)  
                    memory_train_embeddings = torch.cat([memory_train_embeddings[i].unsqueeze(0), memory_train_embeddings2[i].unsqueeze(0)], dim=1)
                    memory_train_targets = torch.cat([memory_train_targets[i].unsqueeze(0), memory_train_targets2[i].unsqueeze(0)+args.num_way], dim =1)
                    memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, 2*args.num_way)
                    memory_test_embeddings = torch.cat([memory_test_embeddings[i].unsqueeze(0), memory_test_embeddings2[i].unsqueeze(0)], dim = 1)
                    memory_test_targets = torch.cat([memory_test_targets[i].unsqueeze(0), memory_test_targets2[i].unsqueeze(0)+args.num_way], dim =1)
                    memory_loss += 1e-4*prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)

                
        param_list = []
        param_names = []
        for name, v in self.model.named_parameters():
            if 'domain_out' not in name: 
                if v.requires_grad:
                    param_list.append(v)
                    param_names.append(name)
        val_grad = torch.autograd.grad(memory_loss, param_list)
        
        val_graddict = {}
        layer_name = []
        for gradient, name in zip(val_grad, param_names):
            split_name = name.split('.')
            layer = split_name[0]
            if layer not in self.args.layer_filters:
                if layer not in layer_name:
                    layer_name.append(layer)
                    val_graddict[layer] = []
                    val_graddict[layer].append(gradient.view(-1))
                else:
                    val_graddict[layer].append(gradient.view(-1))
            else:
                layer_sub = layer+'.'+split_name[1]+'.'+split_name[2]
                if layer_sub not in layer_name:
                    layer_name.append(layer_sub)
                    val_graddict[layer_sub] = []
                    val_graddict[layer_sub].append(gradient.view(-1))
                else:
                    val_graddict[layer_sub].append(gradient.view(-1))

        for key in val_graddict:
            val_graddict[key] = torch.cat(val_graddict[key])
        self.model.zero_grad()
        memory_loss.detach_()
        return val_graddict

    
    def save(self, epoch):
        if self.args.output_folder is not None:
            filename = os.path.join(self.filepath, 'epoch{0}.pt'.format(epoch))
            with open(filename, 'wb') as f:
                state_dict = self.model.state_dict()
                torch.save(state_dict, f)

    def load(self, epoch):
        filename = os.path.join(self.filepath,  'epoch{0}.pt'.format(epoch))
        print('loading model filename', filename)
        self.model.load_state_dict(torch.load(filename))
 

    def valid(self, dataloader_dict, domain_id, epoch):
        self.model.eval()
        acc_dict = {}
        acc_list = []
        for dataname, dataloader in dataloader_dict.items():
            with torch.no_grad():
                with tqdm(dataloader, total=self.args.num_valid_batches) as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        self.model.zero_grad()
                        train_inputs, train_targets = batch['train']
                        train_inputs = train_inputs.to(device=self.args.device)
                        train_targets = train_targets.to(device=self.args.device)
                        if train_inputs.size(2) == 1:
                            train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)

                        train_embeddings = self.model(train_inputs, domain_id)
                        test_inputs, test_targets = batch['test']
                        test_inputs = test_inputs.to(device=self.args.device)
                        test_targets = test_targets.to(device=self.args.device)
                        if test_inputs.size(2) == 1:
                            test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                        test_embeddings = self.model(test_inputs, domain_id)

                        prototypes = get_prototypes(train_embeddings, train_targets, self.args.num_way)
                        accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
                        acc_list.append(accuracy.cpu().data.numpy())
                        pbar.set_description('dataname {} accuracy ={:.4f}'.format(dataname, np.mean(acc_list)))
                        if batch_idx >= self.args.num_valid_batches:
                            break

            avg_accuracy = np.round(np.mean(acc_list), 4)
            acc_dict = {dataname:avg_accuracy}
            return acc_dict
        
    def update_lr(self, domain_id, lr=None):
        params_dict = []
        if domain_id==0:
            layer_params = {}
            layer_name = []
            fast_parameters = []
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    if 'conv' in name:
                        split_name = name.split('.')
                        layer = split_name[0]
                        if layer not in self.args.layer_filters:
                            if layer not in layer_name:
                                layer_name.append(layer)
                                layer_params[layer] = []
                                layer_params[layer].append(p)
                            else:
                                layer_params[layer].append(p)

                        else:
                            layer_sub = layer+'.'+split_name[1]+'.'+split_name[2]
                            if layer_sub not in layer_name:
                                layer_name.append(layer_sub)
                                layer_params[layer_sub] = []
                                layer_params[layer_sub].append(p)
                            else:
                                layer_params[layer_sub].append(p)

                    else:
                        fast_parameters.append(p)

            params_list = []
            for key in layer_params:
                params_list.append({'params':layer_params[key], 'lr':self.init_lr})
            params_list.append({'params':fast_parameters, 'lr':self.init_lr})
            self.optimizer = torch.optim.Adam(params_list, lr=self.init_lr)
        else:
            layer_params = {}
            layer_name = []
            fast_parameters = []
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    if 'conv' in name:
                        split_name = name.split('.')
                        layer = split_name[0]
                        if layer not in self.args.layer_filters:
                            if layer not in layer_name:
                                layer_name.append(layer)
                                layer_params[layer] = []
                                layer_params[layer].append(p)
                            else:
                                layer_params[layer].append(p)

                        else:
                            layer_sub = layer+'.'+split_name[1]+'.'+split_name[2]
                            if layer_sub not in layer_name:
                                layer_name.append(layer_sub)
                                layer_params[layer_sub] = []
                                layer_params[layer_sub].append(p)
                            else:
                                layer_params[layer_sub].append(p)
                    else:
                        fast_parameters.append(p)

            params_list = []
            for key in layer_params:
                params_list.append({'params':layer_params[key], 'lr':lr})
            params_list.append({'params':fast_parameters, 'lr':self.init_lr})
            self.optimizer = torch.optim.Adam(params_list, lr=self.init_lr)

      
def memory(args, dataloader_list):
        memory_data = []
        for dataloader_dict in dataloader_list:
            for dataname, dataloader in dataloader_dict.items():
                memorylist = []
                with tqdm(dataloader, total=args.num_memory_batches) as pbar:
                        for batch_idx, batch in enumerate(pbar):
                            memorylist.append(batch)
                            if batch_idx >= args.num_memory_batches:
                                break
                memory_data.append({dataname:memorylist})
        return memory_data

def main(args):

    train_loader_list, valid_loader_list, test_loader_list = dataset(args, datanames)
    model = PrototypicalNetworkhead1(3,
                                args.embedding_size,
                                hidden_size=args.hidden_size, num_tasks=len(datanames), num_block = args.num_block)
    model.to(device=args.device)

    num_data = len(train_loader_list)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    memorydata = memory(args, train_loader_list)
    each_epoch = args.num_epoch
    seqmeta = SequentialMeta(model, args=args)
    for loaderindex, train_loader in enumerate(train_loader_list):
        model.set_req_grad(loaderindex, False)
        seqmeta.update_lr(loaderindex, lr=1e-3)
        for epoch in range(each_epoch*loaderindex, each_epoch*(loaderindex+1)):
            print('Epoch {}'.format(epoch))
            if loaderindex  == 0:
                memory_train = None
            else:
                if epoch %each_epoch == 0:
                    memory_train = memorydata[:loaderindex]
            seqmeta.train(epoch, train_loader, memory_train, domain_id = loaderindex)
            epoch_acc = []
            for index, test_loader in enumerate(test_loader_list[:loaderindex+1]):
                test_accuracy_dict = seqmeta.valid(test_loader, domain_id = index, epoch = epoch)
                epoch_acc.append(test_accuracy_dict)
            seqmeta.save(epoch)

        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Sequential domain meta learning')
    parser.add_argument('--data_path', type=str, default='',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--output_folder', type=str, default='output/datasset/',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--MiniImagenet_batch_size', type=int, default=3,
        help='Number of tasks in a mini-batch of tasks for MiniImagenet (default: 4).')
    parser.add_argument('--CIFARFS_batch_size', type=int, default=3,
        help='Number of tasks in a mini-batch of tasks for CIFARFS (default: 4).')
    parser.add_argument('--CUB_batch_size', type=int, default=3,
        help='Number of tasks in a mini-batch of tasks for CUB (default: 4).')
    parser.add_argument('--Omniglot_batch_size', type=int, default=3,
        help='Number of tasks in a mini-batch of tasks for Omniglot (default: 4).')
    parser.add_argument('--Aircraft_batch_size', type=int, default=3,
        help='Number of tasks in a mini-batch of tasks for Aircraft (default: 4).')
    parser.add_argument('--num_block', type=int, default=4,
        help='Number of convolution block.')
    parser.add_argument('--num-batches', type=int, default=200,
        help='Number of batches the prototypical network is trained over (default: 200).')
    parser.add_argument('--num_valid_batches', type=int, default=150,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--num_memory_batches', type=int, default=1,
        help='Number of batches the model is trained over (default: 1).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--num_query', type=int, default=10,
        help='Number of query examples per class (k in "k-query", default: 10).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')
    parser.add_argument('--num_epoch', type=int, default=40,
        help='Number of epochs for meta train.') 
    parser.add_argument('--valid_batch_size', type=int, default=5,
        help='Number of tasks in a mini-batch of tasks for validation (default: 5).')
    parser.add_argument('--clip_hyper', type=float, default=10.0)
    parser.add_argument('--LR', type=float, default=2.0)
    parser.add_argument('--hyper-lr', type=float, default=1e-4)
    parser.add_argument('--layer_filters', type=int, nargs='+', default=['conv1', 'conv2', 'conv3', 'conv4'], help='0 = CPU.')
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main(args)

