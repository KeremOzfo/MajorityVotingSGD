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
import random


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


def train_vanilla(args, device):
    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)

    net_users = [get_net(args).to(device) for u in range(num_client)]

    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4) for cl in
                  range(num_client)]
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
    randomWorkers = None
    for p in net_ps.named_parameters():
        names = p[0]
        layer_types.append(names.split('.'))
    errors = []
    worker_masks = []
    accuracys = []
    currentLR = args.lr
    for cl in range(num_client):
        errors.append(torch.zeros(modelsize).to(device))
        worker_masks.append(torch.zeros(modelsize).to(device))
    runs = math.ceil(N_s / (args.bs * num_client))

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    trun = 0
    topk = math.ceil(modelsize / args.majority_Sparsity)
    majority_pool = torch.zeros(modelsize).to(device)
    majority_mask = torch.zeros(modelsize).to(device)

    for epoch in tqdm(range(args.num_epoch)):
        atWarmup = (args.warmUp and epoch < 5)
        if args.warmUp and epoch < 5:
            sf.lr_warm_up(optimizers, args.num_client, epoch, args.lr)

        if epoch in args.lr_change:
            for cl in range(num_client):
                sf.adjust_learning_rate(optimizers[cl], epoch, args.lr_change, args.lr)
        currentLR = sf.get_LR(optimizers[0])

        for run in range(runs):
            for cl in range(num_client):

                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader: ## Train
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizers[cl].zero_grad()
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    optimizers[cl].step()
                    break
            trun += 1 ## Local Iter Counter
            atCommRound = ((args.LocalSGD and trun == args.LSGDturn) or not args.LocalSGD) ## check whether this is the communication round
            if atWarmup: ## warm up phase
                ps_model_flat = sf.get_model_flattened(net_ps, device)
                ps_model_dif = torch.zeros_like(ps_model_flat).to(device)
                for cl in range(num_client): ## get average of all workers' model difference
                    model_flat = sf.get_model_flattened(net_users[cl], device)
                    ps_model_dif.add_(1 / num_client, model_flat.sub(1,ps_model_flat))
                ps_model_flat.add_(1, ps_model_dif) ## update ps_model
                sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
                [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)] ## update worker models
                trun = 0
            elif not atWarmup and atCommRound: ## communication Round
                majority_pool *= 0 ## reset majority pool
                clone_mask = torch.clone(majority_mask).to(device) ## reserve the previous iter mask
                ps_model_flat = sf.get_model_flattened(net_ps, device)
                ps_model_dif = torch.zeros_like(ps_model_flat).to(device)
                difmodels = []
                # wtow_corrs = [] ## stats for wlocal_wlocal
                # wto_prevmask_corrs = [] ## stats for wlocal_to_prevMask
                # new_w_to_newmask = []
                for cl in range(num_client):
                    model_flat = sf.get_model_flattened(net_users[cl], device)
                    difmodel = (model_flat.sub(1,ps_model_flat)).to(device) ### get new difference of the local worker after after errorCor
                    if not args.LocalSGD: ## error correction
                        if epoch in args.lr_change and run == 0:
                            difmodel.add_(errors[cl].mul_(0.1))
                        else:
                            difmodel.add_(errors[cl])
                    else: ## Error Correction if LocalSGD is on
                        if epoch in args.lr_change and run < trun:
                            difmodel.add_(errors[cl].mul_(0.1))
                        else:
                            difmodel.add_(errors[cl])

                    ###Stats##
                    # worker_mask = torch.zeros(modelsize).to(device) ## worker_mask for stat
                    # worker_mask[torch.topk(difmodel.abs(),k=topk,dim=0)[1]] = 1 ## make workermask
                    # wto_prevmask_corrs.append(torch.sum(majority_mask.mul(worker_mask)) * 100 /topk)
                    # wtow_corrs.append(torch.sum(worker_masks[cl].mul(worker_mask)) * 100 / topk)
                    # worker_masks[cl] = worker_mask
                    ##Stats###

                    difmodels.append(difmodel) ## store worker's model Differences
                    majority_pool[torch.topk(difmodel.abs(), k=topk, dim=0)[1]] += 1 ## Collect all topk values into a pool
                #print(majority_pool)
                majority_mask.mul_(0) ## Reset the mask
                majority_mask[torch.topk(majority_pool, k=topk, dim=0)[1]] = 1 ## make new mask from the pool
                # for cl in range(num_client):
                #     new_w_to_newmask.append(torch.sum(worker_masks[cl].mul(majority_mask))*100 /topk)

                ###STAT outputs###
                # print(new_w_to_newmask)
                # print('w_to_prevm')
                # print(wto_prevmask_corrs)
                # print('w_to_w')
                # print(wtow_corrs)
                # print('mask_to_mask')
                # corr2 = torch.sum(majority_mask.mul(clone_mask))
                # print(corr2 * 100 /topk)
                #####

                for cl in range(num_client):
                    if args.quantization:
                        Q_model = sf.quantize(difmodels[cl].mul(majority_mask), args, device)
                        errors[cl] = difmodels[cl].sub(1, Q_model)
                        ps_model_dif.add_(1 / num_client, Q_model)
                    else:
                        errors[cl] = difmodels[cl].mul(1 - majority_mask)
                        ps_model_dif.add_(1 / num_client, difmodels[cl].mul(majority_mask))
                ps_model_flat.add_(1, ps_model_dif) ## update model of the ps
                sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
                [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)] ### update local woker model's
                trun = 0 ## reset localTurn

            # if run % 25 == 0:  ##debug
            #     acc = evaluate_accuracy(net_ps, testloader, device)
            #     print('debug accuracy:', acc * 100)

        acc = evaluate_accuracy(net_ps, testloader, device)
        accuracys.append(acc * 100)
        print('accuracy:', acc * 100, )
    return accuracys

def train_add_drop(args, device):
    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)

    net_users = [get_net(args).to(device) for u in range(num_client)]

    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4) for cl in
                  range(num_client)]
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
    randomWorkers = None
    for p in net_ps.named_parameters():
        names = p[0]
        layer_types.append(names.split('.'))
    errors = []
    workerTops = []
    workerMasks = []
    accuracys = []
    currentLR = args.lr
    for cl in range(num_client):
        errors.append(torch.zeros(modelsize).to(device))
        workerTops.append(torch.zeros(modelsize).to(device))
    runs = math.ceil(N_s / (args.bs * num_client))

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    trun = 0
    topk = math.ceil(modelsize / args.majority_Sparsity)
    majority_pool = torch.zeros(modelsize).to(device)
    majority_mask = torch.zeros(modelsize).to(device)
    first_mask_flag = True

    for epoch in tqdm(range(args.num_epoch)):
        atWarmup = (args.warmUp and epoch < 5)
        if atWarmup:
            sf.lr_warm_up(optimizers, args.num_client, epoch, args.lr)

        if epoch in args.lr_change:
            for cl in range(num_client):
                sf.adjust_learning_rate(optimizers[cl], epoch, args.lr_change, args.lr)
        currentLR = sf.get_LR(optimizers[0])

        for run in range(runs):
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
            trun += 1
            atCommRound = ((args.LocalSGD and trun == args.LSGDturn) or not args.LocalSGD)

            if atWarmup:
                ps_model_flat = sf.get_model_flattened(net_ps, device)
                ps_model_dif = torch.zeros_like(ps_model_flat).to(device)
                for cl in range(num_client):
                    model_flat = sf.get_model_flattened(net_users[cl], device)
                    ps_model_dif.add_(1 / num_client, model_flat.sub(1, ps_model_flat))
                ps_model_flat.add_(1, ps_model_dif)
                sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
                [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
                trun = 0

            elif atCommRound and not atWarmup:
                majority_pool *= 0
                difmodels = []
                ps_model_flat = sf.get_model_flattened(net_ps, device)
                ps_model_dif = torch.zeros_like(ps_model_flat).to(device)
                if first_mask_flag:
                    for cl in range(num_client):
                        workerMask = torch.zeros(modelsize).to(device)
                        model_flat = sf.get_model_flattened(net_users[cl], device)
                        difmodel = model_flat.sub(1,ps_model_flat)
                        workerMask[torch.topk(difmodel.abs(),k=topk,dim=0)[1]] = 1
                        workerMasks.append(workerMask)
                        difmodels.append(difmodel)
                        majority_pool[torch.topk(difmodel.abs(),k=topk,dim=0)[1]] +=1
                    majority_mask[torch.topk(majority_pool,k=topk,dim=0)[1]] = 1
                    first_mask_flag = False
                else:
                    trun = 0
                    for cl in range(num_client):
                        model_flat = sf.get_model_flattened(net_users[cl], device)
                        difmodel = (model_flat.sub(1,ps_model_flat))
                        if not args.LocalSGD:  ## error correction
                            if epoch in args.lr_change and run == 0:
                                difmodel.add_(errors[cl].mul_(0.1))
                            else:
                                difmodel.add_(errors[cl])
                        else:  ## Error Correction if LocalSGD is on
                            if epoch in args.lr_change and run < trun:
                                difmodel.add_(errors[cl].mul_(0.1))
                            else:
                                difmodel.add_(errors[cl])
                        difmodels.append(difmodel)
                        workerMasks[cl] = sf.worker_voting2(workerMasks[cl], difmodel, args, device)


                    for worker_mask in workerMasks:
                        majority_pool.add_(1, worker_mask)
                    majority_mask.mul_(0)
                    majority_mask[torch.topk(majority_pool, k=topk, dim=0)[1]] = 1
                for cl in range(num_client):
                    if args.quantization:
                        Q_model = sf.quantize(difmodels[cl].mul(majority_mask),args,device)
                        errors[cl] = difmodels[cl].sub(1,Q_model)
                        ps_model_dif.add_(1/num_client, Q_model)
                    else:
                        errors[cl] = difmodels[cl].mul(1-majority_mask)
                        ps_model_dif.add_(1 / num_client, difmodels[cl].mul(majority_mask))
                ps_model_flat.add_(1, ps_model_dif)
                sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
                [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
                trun = 0


            # if run % 25 == 0:  ##debug
            #     acc = evaluate_accuracy(net_ps, testloader, device)
            #     print('debug accuracy:', acc * 100)

        acc = evaluate_accuracy(net_ps, testloader, device)
        accuracys.append(acc * 100)
        print('accuracy:', acc * 100, )
    return accuracys

def train_random(args, device):
    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)

    net_users = [get_net(args).to(device) for u in range(num_client)]

    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4) for cl in
                  range(num_client)]
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
    randomWorkers = None
    for p in net_ps.named_parameters():
        names = p[0]
        layer_types.append(names.split('.'))
    errors = []
    accuracys = []
    currentLR = args.lr
    for cl in range(num_client):
        errors.append(torch.zeros(modelsize).to(device))
    runs = math.ceil(N_s / (args.bs * num_client))

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    trun = 0
    topk = math.ceil(modelsize / args.majority_Sparsity)
    majority_pool = torch.zeros(modelsize).to(device)
    majority_mask = torch.ones(modelsize).to(device)

    for epoch in tqdm(range(args.num_epoch)):
        atWarmup = (args.warmUp and epoch < 5)
        if args.warmUp and epoch < 5:
            sf.lr_warm_up(optimizers, args.num_client, epoch, args.lr)

        if epoch in args.lr_change:
            for cl in range(num_client):
                sf.adjust_learning_rate(optimizers[cl], epoch, args.lr_change, args.lr)
        currentLR = sf.get_LR(optimizers[0])

        for run in range(runs):
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
            trun += 1
            atCommRound = ((args.LocalSGD and trun == args.LSGDturn) or not args.LocalSGD)
            if atWarmup:
                ps_model_flat = sf.get_model_flattened(net_ps, device)
                ps_model_dif = torch.zeros_like(ps_model_flat).to(device)
                for cl in range(num_client):
                    model_flat = sf.get_model_flattened(net_users[cl], device)
                    ps_model_dif.add_(1 / num_client, model_flat.sub(1,ps_model_flat))
                ps_model_flat.add_(1, ps_model_dif)
                sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
                [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
                trun = 0
            elif not atWarmup and atCommRound:
                majority_mask.mul_(0)
                majority_pool.mul_(0)
                ps_model_flat = sf.get_model_flattened(net_ps, device)
                ps_model_dif = torch.zeros_like(ps_model_flat).to(device)
                difmodels = []
                for cl in range(num_client):
                    model_flat = sf.get_model_flattened(net_users[cl], device)
                    difmodel = (model_flat.sub(1, ps_model_flat)).to(
                        device)  ### get new difference of the local worker after after errorCor
                    if not args.LocalSGD:  ## error correction
                        if epoch in args.lr_change and run == 0:
                            difmodel.add_(errors[cl].mul_(0.1))
                        else:
                            difmodel.add_(errors[cl])
                    else:  ## Error Correction if LocalSGD is on
                        if epoch in args.lr_change and run < trun:
                            difmodel.add_(errors[cl].mul_(0.1))
                        else:
                            difmodel.add_(errors[cl])
                    difmodels.append(difmodel)
                    w_inds = (torch.topk(difmodel.abs(), k=topk, dim=0)[1])
                    majority_pool[w_inds] += 1
                non_zero_Inds = majority_pool.nonzero(as_tuple=False)
                non_zero_cumsum = majority_pool[non_zero_Inds].cumsum(dim=0)
                start = non_zero_cumsum[0].int()
                end = non_zero_cumsum[non_zero_cumsum.numel()-1].int()
                while torch.sum(majority_mask) < topk: ## gotta optimize
                    r = random.randint(start,end)
                    selectedInd = torch.where(non_zero_cumsum >= r)[0][0]
                    majority_mask[non_zero_Inds[selectedInd]] = 1
                for cl in range(num_client):
                    ps_model_dif.add_(1 / num_client, difmodels[cl].mul(majority_mask))
                    errors[cl] = difmodels[cl].mul(1 - majority_mask)
                ps_model_flat.add_(1, ps_model_dif)
                sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
                [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
                trun = 0

            # if run % 25 == 0:  ##debug
            #     acc = evaluate_accuracy(net_ps, testloader, device)
            #     print('debug accuracy:', acc * 100)

        acc = evaluate_accuracy(net_ps, testloader, device)
        accuracys.append(acc * 100)
        print('accuracy:', acc * 100, )
    return accuracys

