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

    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4) for cl in range(num_client)]
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
    runs = math.ceil(N_s/(args.bs*num_client))

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    majority_mask = torch.zeros(modelsize).to(device)
    new_majority_mask = torch.zeros(modelsize).to(device)
    for epoch in tqdm(range(args.num_epoch)):
        if epoch == args.errDecayVals[0] and args.errorDecay is True:
            errorCorCof = args.errDecayVals[1]
        if args.warmUp and epoch < 5:
            sf.lr_warm_up(optimizers, args.num_client, epoch, args.lr)

        if epoch in args.lr_change:
            for cl in range(num_client):
                sf.adjust_learning_rate(optimizers[cl], epoch, args.lr_change, args.lr)
        currentLR = sf.get_LR(optimizers[0])

        for run in range(runs):
            majority_pool = torch.zeros(modelsize).to(device)
            add_pool = torch.zeros(modelsize).to(device)
            drop_pool = torch.zeros(modelsize).to(device)
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
            if args.RandomWorkers[0]:
                randomWorkers = np.random.choice(range(num_client), args.RandomWorkers[1], replace=False)

            if not (args.warmUp and epoch < 5):
                for cl in range(num_client):
                    grad_flat = sf.get_grad_flattened(net_users[cl], device)
                    grad_flat.add_(errors[cl] * errorCorCof)

                    if args.AddDrop and (run > 1 or epoch>5):
                        w_add , w_drop = sf.worker_voting(majority_mask, grad_flat, args.sparsity_window, args.K_val, device)
                        add_pool[w_add] += 1
                        drop_pool[w_drop] += 1

                    else:
                        if args.RandomWorkers[0]:
                            if cl in randomWorkers:
                                sf.collectMajority_grad(grad_flat,majority_pool,args.layerMajority,args.majority_Sparsity
                                                    , ind_pairs,args.RandomSelect,args.type ,device)
                            break
                        else:
                            sf.collectMajority_grad(grad_flat, majority_pool, args.layerMajority, args.majority_Sparsity
                                                        , ind_pairs, args.RandomSelect, args.type, device)

                if args.AddDrop and (run > 1 or epoch>5):
                    k = math.ceil(modelsize/args.K_val)
                    #print(torch.sum(majority_mask),'prevMask')
                    dropInds = torch.topk(drop_pool, k=k, dim=0)[1]
                    new_majority_mask[dropInds] = 0
                    #print(torch.sum(majority_mask),'drops')
                    addInds = torch.topk(add_pool, k=k, dim=0)[1]
                    new_majority_mask[addInds] = 1
                    #print(torch.sum(majority_mask),'adds',torch.numel(addInds))

                else:
                    majority_mask = torch.zeros(modelsize).to(device)
                    topk = math.ceil(modelsize/args.sparsity_window)
                    inds = torch.topk(majority_pool, k= topk,dim=0)[1]
                    majority_mask[inds] = 1
                errorMask = 1 - majority_mask
                for cl in range(num_client):
                    grad_flat = sf.get_grad_flattened(net_users[cl], device)
                    grad_flat.mul_(majority_mask)
                    errors[cl] = grad_flat.mul(errorMask)
                    sf.make_grad_unflattened(net_users[cl], grad_flat, net_sizes, ind_pairs)



            sf.zero_grad_ps(net_ps)
            [sf.push_grad(net_users[cl], net_ps, num_client) for cl in range(num_client)]
            if args.AddDrop:
                if (args.warmUp and epoch==4 and run==runs-1) or (not args.warmUp and run ==0 and epoch==0):
                    k = math.ceil(modelsize / args.sparsity_window)
                    tempInd = torch.topk((sf.get_grad_flattened(net_ps, device)).abs(), k=k, dim=0)[1]
                    majority_mask[tempInd] = 1
                    new_majority_mask = majority_mask
                else:
                    majority_mask = new_majority_mask
            sf.update_model(net_ps, lr=currentLR)
            [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]


            # if run % 25 == 0: ##debug
            #     acc = evaluate_accuracy(net_ps, testloader, device)
            #     print('debug accuracy:', acc * 100)


        acc = evaluate_accuracy(net_ps, testloader, device)
        accuracys.append(acc * 100)
        print('accuracy:',acc*100,)
    return accuracys