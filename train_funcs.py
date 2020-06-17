import torch
from torch.utils.data import DataLoader
# custom modules
import data_loader as dl
from nn_classes import *
import server_functions as sf
import math
from parameters import *
import time
import numpy as np
from tqdm import tqdm


def evaluate_accuracy(model, testloader, device):
    """Calculates the accuracy of the model"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)


    net_users = [get_net(args).to(device) for u in range(num_client)]

    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr) for cl in range(num_client)]
    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    modelsize = sf.count_parameters(net_ps)
    errorCorCof = 1
    layer_types = []
    for p in net_ps.named_parameters():
        names = p[0]
        layer_types.append(names.split('.'))
    errors = []
    accuracys = []
    currentLR = args.lr
    for cl in range(num_client):
        errors.append(torch.zeros(modelsize).to(device))
    runs = math.ceil(N_s/(args.bs*num_client))

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    for epoch in tqdm(range(args.num_epoch)):
        if epoch == args.errDecayVals[0] and args.errorDecay is True:
            errorCorCof = args.errDecayVals[1]
        if args.warmUp and epoch < 6:
            for cl in range(num_client):
                for param_group in optimizers[cl].param_groups:
                    if epoch ==0:
                        param_group['lr'] = 0.1
                    else:
                        lr_change = (args.lr - 0.1) / 5
                        param_group['lr'] = (lr_change * epoch) + 0.1
        if epoch in args.lr_change:
            for cl in range(num_client):
                sf.adjust_learning_rate(optimizers[cl], epoch, args.lr_change, args.lr)
            currentLR = sf.get_LR(epoch, args.lr, args.lr_change)

        for run in range(runs):
            majority_pool = torch.zeros(modelsize)
            for cl in range(num_client):

                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizers[cl].zero_grad()
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    optimizers[cl].step()
                    break
            ps_model_flat = sf.get_model_flattened(net_ps, device)
            ps_model_dif = torch.zeros_like(ps_model_flat)
            for cl in range(num_client):
                with torch.no_grad():
                    model_flat = sf.get_model_flattened(net_users[cl], device)
                    difmodel = (model_flat.sub(ps_model_flat)).to(device)
                    difmodel.add_(errors[cl] * currentLR * errorCorCof)

                if not (args.warmUp and epoch < 6):
                    sf.collectMojority(difmodel,majority_pool,1000,20,ind_pairs,5,device)

            majority_mask = torch.zeros_like(ps_model_flat).to(device)

            if not (args.warmUp and epoch < 6):
                topk = math.ceil(modelsize/args.sparsity_window)
                vals, inds = torch.topk(majority_pool, k= topk,dim=0)
                majority_mask[inds] = 1
                errorMask = 1 - majority_mask
                for cl in range(num_client):
                    with torch.no_grad():
                        model_flat = sf.get_model_flattened(net_users[cl], device)
                        difmodel = (model_flat.sub(ps_model_flat)).to(device)
                        errors[cl] = (difmodel * errorMask) / currentLR
                        ps_model_dif.add_((difmodel.mul_(majority_mask)) / num_client)
            else:
                for cl in range(num_client):
                    model_flat = sf.get_model_flattened(net_users[cl], device)
                    difmodel = (model_flat.sub(ps_model_flat)).to(device)
                    ps_model_dif.add_(difmodel / num_client)
            ps_model_flat.add_(ps_model_dif)

            sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)

            [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]


            if run % 25 == 0: ##debug
                acc = evaluate_accuracy(net_ps, testloader, device)
                print('debug accuracy:', acc * 100)


        acc = evaluate_accuracy(net_ps, testloader, device)
        accuracys.append(acc * 100)
        print('accuracy:',acc*100,)
    return accuracys