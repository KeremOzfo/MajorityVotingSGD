import torchvision
import torchvision.transforms as transforms
import torch

import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

import nn_classes
import data_loader
import ps_functions



# select gpu
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(device)

# number of workers
N_w = 15
# number of training samples
# Cifar10  50,000
# Fashin MNIST 60,000
N_s = 60000
batch = 64
runs = 251
error_correction = True
mode = 'majority_Voting'






lenmodel= nn_classes.MNIST_NET().to(device)
total_params = ps_functions.count_parameters(lenmodel)
iter = 1
sparse_results = []
server_sparse = 1
worker_sparse = 100
spars = int(total_params / worker_sparse)
total_spars = int(spars /server_sparse)
spars_percantage = round(total_spars/total_params *100,2)
for i in range(iter):
    print(i)
    nets = [nn_classes.MNIST_NET().to(device) for n in range(N_w)]
    reserveNets = [nn_classes.MNIST_NET().to(device) for n in range(N_w)]
    netDifs = [nn_classes.MNIST_NET().to(device) for n in range(N_w)]
    errormodels = [nn_classes.MNIST_NET().to(device) for n in range(N_w)]
    ps_model = nn_classes.MNIST_NET().to(device)
    avg_model = nn_classes.MNIST_NET().to(device)
    sparse_dif_model = nn_classes.MNIST_NET().to(device)
    prev_mask = nn_classes.MNIST_NET().to(device)

    trainloaders, testloader = data_loader.MNIST_data(batch, N_w, N_s)

    w_index = 0
    results = []
    lr = 1e-1
    momentum = 0
    weight_decay = 1e-4

    criterions = [nn.CrossEntropyLoss() for n in range(N_w)]
    optimizers = [torch.optim.SGD(nets[n].parameters(), lr=lr, weight_decay= weight_decay, momentum=0) for n in range(N_w)]


    # initilize all weights equally

    [ps_functions.synch_weight(nets[i], ps_model) for i in range(N_w)]
    [ps_functions.synch_weight(reserveNets[i], ps_model) for i in range(N_w)]
    [ps_functions.synch_weight(netDifs[i], ps_model) for i in range(N_w)]
    [ps_functions.synch_weight(errormodels[i], ps_model) for i in range(N_w)]
    ps_functions.synch_weight(avg_model, ps_model)
    ps_functions.synch_weight(sparse_dif_model, ps_model)
    ps_functions.synch_weight(prev_mask,ps_model)

    ps_functions.initialize_zero(prev_mask)
    ps_functions.initialize_zero(avg_model)
    for i in range(N_w):
        ps_functions.initialize_zero(errormodels[i])
    for r in tqdm(range(runs)):
        # index of the worker doing local SGD
        w_index = w_index % N_w
        for worker in range(N_w):
            for data in trainloaders[worker]:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizers[worker].zero_grad()
                preds = nets[worker](inputs)
                loss = criterions[worker](preds, labels)
                loss.backward()
                optimizers[worker].step()
                break
        if r ==0:
            for worker in range(N_w):
                ps_functions.weight_accumulate(nets[worker],avg_model,N_w)
        else:
            for worker in range(N_w):
                ps_functions.weight_dif(nets[worker], avg_model, netDifs[worker])
                if error_correction:
                    ps_functions.weight_accumulate(errormodels[worker],nets[worker],1)
        ps_functions.make_sparse_diff(spars,server_sparse, netDifs, N_w, sparse_dif_model, device, errormodels,mode)
        ps_functions.weight_accumulate(sparse_dif_model,avg_model,1)
        for worker in  range(N_w):
            ps_functions.synch_weight(nets[worker],avg_model)


        if r % 10 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = avg_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
            results.append(100 * correct / total)
        # moving to next worker
        w_index += 1
    sparse_results.append(results)
np.save('sparse'+str(spars_percantage)+'_type-' + mode +'-ErrorCor-'+str(error_correction)+'-new', sparse_results)




