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
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total, loss



def train(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]
    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4) for cl in
                  range(num_client)]
    schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[cl], milestones=args.lr_change, gamma=0.1) for cl in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    accuracys = []
    losses =[]
    runs = math.ceil(N_s / (args.bs * num_client * args.LocalIter))
    acc, loss = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    losses.append(loss)
    assert N_s/num_client > args.LocalIter * args.bs

    for epoch in tqdm(range(args.num_epoch)):
        atWarmup = (args.warmUp and epoch < 5)
        if atWarmup:
            sf.lr_warm_up(optimizers, args.num_client, epoch, args.lr)

        for run in range(runs):
            for cl in range(num_client):
                localIter = 0

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
                    localIter +=1
                    if localIter == args.LocalIter:
                        break
            weight_vec = [args.alfa for cl in range(num_client)]## subject of change
            if atWarmup:
                weight_vec = np.asarray(weight_vec)
                weight_vec[:] =1
            ps_model_flat =sf.get_model_flattened(net_ps, device)
            worker_model_diffs = [(sf.get_model_flattened(net_users[cl],device)).sub(1,ps_model_flat) for cl in range(num_client)]
            avg_model = torch.zeros_like(ps_model_flat)
            for cl in range(num_client):
                avg_model.add_(1/num_client, worker_model_diffs[cl])
            ps_model_flat.add_(1,avg_model)
            sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
            for cl in range(num_client):
                worker_model = ps_model_flat
                worker_model.add_(1-weight_vec[cl],worker_model_diffs[cl])
                worker_model.add_(weight_vec[cl], avg_model)
                sf.make_model_unflattened(net_users[cl], worker_model, net_sizes, ind_pairs)

        acc, loss = evaluate_accuracy(net_ps, testloader, device)
        accuracys.append(acc * 100)
        losses.append(loss)
        print('accuracy:',acc*100)
        print(loss)
        if not atWarmup:
            [schedulers[cl].step() for cl in range(num_client)] ## adjust Learning rate
    return accuracys, losses



