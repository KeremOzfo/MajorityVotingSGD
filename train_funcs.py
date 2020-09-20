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
                majority_pool *= 0
                clone_mask = torch.clone(majority_mask).to(device)
                ps_model_flat = sf.get_model_flattened(net_ps, device)
                ps_model_dif = torch.zeros_like(ps_model_flat).to(device)
                difmodels = []
                wtow_corrs = []
                wto_prevmask_corrs = []
                for cl in range(num_client):
                    model_flat = sf.get_model_flattened(net_users[cl], device)
                    if not args.LocalSGD:
                        if epoch in args.lr_change and run == 0:
                            model_flat.add_(errors[cl].mul_(0.1))
                        else:
                            model_flat.add_(errors[cl])
                    else:
                        if epoch in args.lr_change and run < trun:
                            model_flat.add_(errors[cl].mul_(0.1))
                        else:
                            model_flat.add_(errors[cl])
                    difmodel = (model_flat.sub(1,ps_model_flat)).to(device)
                    worker_mask = torch.zeros(modelsize).to(device)

                    worker_mask[torch.topk(difmodel.abs(),k=topk,dim=0)[1]] = 1
                    wto_prevmask_corrs.append(torch.sum(majority_mask.mul(worker_mask)) * 100 /topk)
                    wtow_corrs.append(torch.sum(worker_masks[cl].mul(worker_mask)) * 100 / topk)

                    worker_masks[cl] = worker_mask
                    difmodels.append(difmodel)
                    majority_pool[torch.topk(difmodel.abs(), k=topk, dim=0)[1]] += 1
                majority_mask.mul_(0)
                majority_mask[torch.topk(majority_pool, k=topk, dim=0)[1]] = 1
                print('w_to_prevm')
                print(wto_prevmask_corrs)
                print('w_to_w')
                print(wtow_corrs)
                print('mask_to_mask')
                corr2 = torch.sum(majority_mask.mul(clone_mask))
                print(corr2 * 100 /topk)
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
                        model_flat = sf.get_model_flattened(net_users[cl], device)
                        difmodel = model_flat.sub(1,ps_model_flat)
                        difmodels.append(difmodel)
                        majority_pool[torch.topk(difmodel.abs(),k=topk,dim=0)[1]] +=1
                    majority_mask[torch.topk(majority_pool,k=topk,dim=0)[1]] = 1
                    first_mask_flag = False
                else:
                    addInds = []
                    dropInds = []
                    for cl in range(num_client):
                        model_flat = sf.get_model_flattened(net_users[cl], device)
                        if not args.LocalSGD:
                            if epoch in args.lr_change and run == 0:
                                model_flat.add_(errors[cl].mul_(0.1))
                            else:
                                model_flat.add_(errors[cl])
                        else:
                            if epoch in args.lr_change and run < trun:
                                model_flat.add_(errors[cl].mul_(0.1))
                            else:
                                model_flat.add_(errors[cl])
                            trun = 0
                        difmodel = (model_flat.sub(1,ps_model_flat))
                        difmodels.append(difmodel)
                        workerTop, worker_add , worker_drop = sf.worker_voting(majority_mask, difmodel, args, device)
                        addInds.append(worker_add)
                        dropInds.append(worker_drop)
                        corr = torch.sum(workerTop.mul(workerTops[cl]))
                        corrVal = corr * 100 / topk
                        print(corrVal)
                        workerTops[cl] = workerTop

                    for addm, dropm in zip(addInds,dropInds):
                        cloneMask = torch.clone(majority_mask).to(device)
                        cloneMask[addm] = 1
                        cloneMask[dropm] = 0
                        majority_pool.add_(1, cloneMask)
                    majority_mask.mul_(0)
                    majority_mask[torch.topk(majority_pool, k=topk, dim=0)[1]] = 1
                for cl in range(num_client):
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
                    if not args.LocalSGD:
                        if epoch in args.lr_change and run == 0:
                            model_flat.add_(errors[cl].mul_(0.1))
                        else:
                            model_flat.add_(errors[cl])
                    else:
                        if epoch in args.lr_change and run < trun:
                            model_flat.add_(errors[cl].mul_(0.1))
                        else:
                            model_flat.add_(errors[cl])
                    difmodel = (model_flat.sub(1,ps_model_flat)).to(device)
                    difmodels.append(difmodel)
                    w_inds = (torch.topk(difmodel.abs(), k=topk, dim=0)[1])
                    majority_pool[w_inds] += 1
                ts = time.time()
                non_zero_Inds = majority_pool.nonzero(as_tuple=False)
                non_zero_cumsum = majority_pool[non_zero_Inds].cumsum(dim=0)
                start = non_zero_cumsum[0].int()
                end = non_zero_cumsum[non_zero_cumsum.numel()-1].int()
                counter = 0
                while torch.sum(majority_mask) < topk: ## gotta optimize
                    counter +=1
                    r = random.randint(start,end)
                    selectedInd = torch.where(non_zero_cumsum >= r)[0][0]
                    majority_mask[non_zero_Inds[selectedInd]] = 1
                #print('t2', time.time() - ts, 'c',counter, 'tk',topk)
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

