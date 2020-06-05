import torch.nn as nn
import numpy as np
import torch
from copy import deepcopy


def synch_weight(model, model_synch):
    for param, param_synch in zip(model.parameters(), model_synch.parameters()):
        param.data = param_synch.data + 0

def weight_dif(model, model_synch,model_dif):
    for param, param_synch, param_dif in zip(model.parameters(), model_synch.parameters(), model_dif.parameters()):
        param_dif.data = param.data - param_synch.data + 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def average_model(net_avg, net_toavg):
    # net_toavg is broadcasted
    # net_avg uses net_toavg to update its parameters
    for param_avg, param_toavg in zip(net_avg.parameters(), net_toavg.parameters()):
        param_avg.data.mul_(0.5)
        param_avg.data.add_(0.5, param_toavg.data)
    return None

def initialize_zero(model):
    for param in model.parameters():
        param.data.mul_(0)
    return None


def weight_accumulate(model, agg_model, num):

    for param, ps_param in zip(model.parameters(), agg_model.parameters()):
        ps_param.data += param.data / num
    return None

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    if epoch > 100:
        lr = round(lr * (0.1 ** (epoch // 75)),3)
    else:
        lr = round(lr * (0.1 ** (epoch // 100)), 3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_sparse_diff(top_k,server_sp, models,num_w, new_model, device, errors,mode): ##Majority Voting and Weighted Majority Voting
    grad_flattened = [torch.empty(0).to(device) for n in range(num_w)]
    top_k_grads = [] # for each worker
    grad_min_values = [] # for each worker
    masks = []
    tops = int(top_k/server_sp)
    mask_flatted = torch.empty(0).to(device)
    top_weights = []
    weighted_array = []
    for w in range(num_w):
        for p in models[w].parameters():
            a = p.data.flatten().to(device)
            grad_flattened[w] = torch.cat((a, grad_flattened[w]), 0)
        top_weights.append(grad_flattened[w].abs().max()) ### top abs weight dif for each worker in array
    top_value = np.max(top_weights) ### top abs dif weight among all workers
    for i in range(4): ### group weights for W-Majority
        weighted_array.append(top_value)
        top_value = top_value / 2
    weighted_array = np.flip(weighted_array) ### groups
    for w in range(num_w):
        top_k_grads.append(torch.topk(grad_flattened[w].abs(), top_k)[0].to(device)) ### topk grads for each worker
        grad_min_values.append(top_k_grads[w].min().to(device)) #mid value for each worker

    for w in range(num_w): ### mask operation, adding up each workers maskt to create one mask for each layer
        for count, p in enumerate(models[w].parameters()):
            if mode =='majority_Voting':
                sparse_mask = p.data.abs() >= grad_min_values[w]
                if w ==0:
                    masks.append(sparse_mask.float())
                else:
                    masks[count].add_(sparse_mask.float())
            else: ### W-Majority
                sparse_mask = p.data.abs() >= grad_min_values[w]
                sparse_mask = sparse_mask.float() # m
                for i in range(len(weighted_array-1)): ### add 1 to all mask values if (ge) of groups
                    sparse_mask.add_((p.data.abs() >= weighted_array[i]).float())

                if w ==0:
                    masks.append(sparse_mask)
                else:
                    masks[count].add_(sparse_mask)
    for i in range(len(masks)): ### flat all masks of the layers
        a = masks[i].flatten().to(device)
        mask_flatted = torch.cat((a,mask_flatted),0)

    top_repeated = torch.topk(mask_flatted,tops)[0].to(device) # top repeated masks (indexes)
    min_repeat = top_repeated.min().to(device)
    initialize_zero(new_model)
    for worker in range(num_w):
        for count, (p, new_p,error_p) in enumerate(zip(models[worker].parameters(),new_model.parameters(),errors[worker].parameters())):
            sparse_mask = masks[count] >= min_repeat
            error_mask = masks[count] < min_repeat
            new_p.data += sparse_mask.float() * (p.data[:] / num_w)
            error_p.data = (error_mask.float() * p.data[:])
    return None



def sparse_extra(top_k,models, prev_mask,num_w,new_model, device, errors, iter):
    second_flat = [torch.empty(0).to(device) for n in range(num_w)]
    additional_weight_number = int (top_k / 10) # worker exlusive spars ratio
    if iter > 1:
        flatz = torch.empty(0).to(device)
        for new_p in new_model.parameters():
           flatz = torch.cat((new_p.data.flatten().to(device),flatz),0)
        top_new_Gradz = (torch.topk(flatz.abs(),top_k)[0]).to(device)
        grad_value = top_new_Gradz.min().to(device)
        initialize_zero(new_model)
        top_indv_grads = []
        for w in range(num_w):
            indv_model = [] # worker model without the topk indexes of broadcasted model
            mask_indv = []
            for p, prev_p in zip(models[w].parameters(), prev_mask.parameters()):
                a = (p.data.sub(p.data * prev_p.data)).to(device) ## previous top weight indexes removed from worker's weights
                indv_model.append(a)
                second_flat[w] = torch.cat((a.flatten(), second_flat[w]), 0)
            top_indv_grads.append((torch.topk(second_flat[w].abs(),additional_weight_number)[0]).to(device)) ## top worker-exclusive weights
            indv_min_value = (top_indv_grads[w].min().to(device))
            for count, (p, prev_p, new_p, error_p) in enumerate(zip(models[w].parameters(), prev_mask.parameters(), new_model.parameters(),errors[w].parameters())):
                additional_mask = (indv_model[count].abs().ge(indv_min_value)).float()
                mask = prev_p.data.add(additional_mask)
                mask_indv.append(mask)
                error_mask = 1-mask
                new_p.data.add_((p.data * mask)[:] /num_w)
                error_p.data = p.data * error_mask
        for prev_p, new_p in zip(prev_mask.parameters(), new_model.parameters()):
            prev_p.data = (new_p.data.abs().ge(grad_value)).float()
    else:
        initialize_zero(new_model)
        for w in range(num_w):
            flatz = torch.empty(0).to(device)
            for p in models[w].parameters():
                flatz = torch.cat((p.data.flatten().to(device), flatz), 0)
            top_new_Gradz = torch.topk(flatz.abs(), top_k)[0].to(device)
            grad_value = top_new_Gradz.min().to(device)
            for p, new_p in zip(models[w].parameters(), new_model.parameters()):
                mask = (p.data.abs().ge(grad_value)).float()
                new_p.data.add_((p.data * mask)[:] / num_w)
        new_model_flatten = torch.empty(0).to(device)
        for new_p in new_model.parameters():
            new_model_flatten = torch.cat((new_p.data.flatten().to(device),new_model_flatten), 0)
        top_new_Gradz_2 = torch.topk(new_model_flatten.abs(),top_k)[0].to(device)
        grad_value_2 = top_new_Gradz_2.min().to(device)
        for prev_p, new_p in zip(prev_mask.parameters(), new_model.parameters()):
            prev_p.data = (new_p.data.ge(grad_value_2)).float()
    return None

def sparse_extra_alt(top_k,models, prev_mask,num_w,new_model, device, errors, iter):
    second_flat = [torch.empty(0).to(device) for n in range(num_w)]
    additional_weight_number = int (top_k / 10) # worker exlusive spars ratio
    initialize_zero(new_model)
    if iter > 1:
        flatz = torch.empty(0).to(device)
        top_indv_grads = []
        for w in range(num_w):
            indv_model = [] # worker model without the topk indexes of broadcasted model
            mask_indv = []
            for p, prev_p in zip(models[w].parameters(), prev_mask.parameters()):
                a = (p.data.sub(p.data * prev_p.data)).to(device) ## previous top weight indexes removed from worker's weights
                indv_model.append(a)
                second_flat[w] = torch.cat((a.flatten(), second_flat[w]), 0)
            top_indv_grads.append((torch.topk(second_flat[w].abs(),additional_weight_number)[0]).to(device)) ## top worker-exclusive weights
            indv_min_value = (top_indv_grads[w].min().to(device))
            for count, (p, prev_p, new_p, error_p) in enumerate(zip(models[w].parameters(), prev_mask.parameters(), new_model.parameters(),errors[w].parameters())):
                additional_mask = (indv_model[count].abs().ge(indv_min_value)).float()
                mask = prev_p.data.add(additional_mask)
                mask_indv.append(mask)
                error_mask = 1-mask
                new_p.data.add_((p.data * mask)[:] /num_w)
                error_p.data = p.data * error_mask
        for new_p in new_model.parameters():
           flatz = torch.cat((new_p.data.flatten().to(device),flatz),0)
        top_new_Gradz = (torch.topk(flatz.abs(),top_k)[0]).to(device)
        grad_value = top_new_Gradz.min().to(device)
        for prev_p, new_p in zip(prev_mask.parameters(), new_model.parameters()):
            prev_p.data = (new_p.data.abs().ge(grad_value)).float()
    else:
        for w in range(num_w):
            flatz = torch.empty(0).to(device)
            for p in models[w].parameters():
                flatz = torch.cat((p.data.flatten().to(device), flatz), 0)
            top_new_Gradz = torch.topk(flatz.abs(), top_k)[0].to(device)
            grad_value = top_new_Gradz.min().to(device)
            for p, new_p in zip(models[w].parameters(), new_model.parameters()):
                mask = (p.data.abs().ge(grad_value)).float()
                new_p.data.add_((p.data * mask)[:] / num_w)
        new_model_flatten = torch.empty(0).to(device)
        for new_p in new_model.parameters():
            new_model_flatten = torch.cat((new_p.data.flatten().to(device),new_model_flatten), 0)
        top_new_Gradz_2 = torch.topk(new_model_flatten.abs(),top_k)[0].to(device)
        grad_value_2 = top_new_Gradz_2.min().to(device)
        for prev_p, new_p in zip(prev_mask.parameters(), new_model.parameters()):
            prev_p.data = (new_p.data.ge(grad_value_2)).float()
    return None

def sparse_group(top_k,groups,g_denominator,models,num_w, new_model, device, errors):
    initialize_zero(new_model)
    grad_flattened = [torch.empty(0).to(device) for n in range(num_w)]
    mask_list =[]
    top_k_grads = []
    top_weights = []
    weighted_arrays =[] ### stores all groupings for individual worker
    avg_list = []
    for w in range(num_w):
        for p in models[w].parameters():
            a = p.data.flatten().to(device)
            grad_flattened[w] = torch.cat((a, grad_flattened[w]), 0)
        top_weights.append(grad_flattened[w].abs().max())
        weighted_array = [] ## group values for individual worker
        w_top_value = 0
        for i in range(groups):
            if i ==0:
                w_top_value = top_weights[w]
            weighted_array.append(w_top_value)
            w_top_value = w_top_value / g_denominator
        weighted_array[groups-1] = 0
        weighted_arrays.append(np.flip(weighted_array)) #
    for w in range(num_w):
        top_k_grad = (torch.topk(grad_flattened[w].abs(), top_k)[0].to(device))
        top_k_grads.append(top_k_grad)
        grad_min_values = (top_k_grad.min().to(device))
        masks = [] ## masks for individual workers
        for p in models[w].parameters():
            sparse_mask = torch.zeros_like(p).data
            sparse_mask_2 = (p.data.abs().ge(grad_min_values)).float()
            for i in range(groups-1):
                sparse_mask.add_((p.data.abs().ge(weighted_arrays[w][i])).float())
            masks.append(sparse_mask * sparse_mask_2)
        mask_list.append(masks)
    for w in range(num_w):
        avgs = [] ## group by group averages of indivual worker
        for i in range(groups-1):
            mask_1 = (top_k_grads[w] >= (weighted_arrays[w][i])).int()
            mask_2 = (top_k_grads[w] <= weighted_arrays[w][i+1]).int()
            mask_1 = mask_1.add_(mask_2)
            mask_1 = mask_1.eq(2)
            selected_groups = top_k_grads[w].masked_select(mask_1)
            if len(selected_groups) > 0:
                avgs.append(selected_groups.mean())
                #print('mean is ',selected_groups.mean())
            else:
                avgs.append(0)
        avg_list.append(avgs)
    initialize_zero(new_model)
    for worker in range(num_w):
        for count, (p, new_p, error_p) in enumerate(
                zip(models[worker].parameters(), new_model.parameters(), errors[worker].parameters())):
            sign_mask = (p.data > 0).float()
            sign_mask.add_(((p.data<0).float()).mul(-1))
            sign_mask = sign_mask.to(device)
            tensors = torch.zeros_like(p.data)
            for i in range(1,groups):
                group_mask = (mask_list[worker][count].eq(i).float()).to(device)
                avg = (avg_list[worker][i-1])
                tensors.add_((group_mask * sign_mask).mul(avg))
            new_p.data.add_(tensors[:] / num_w)
            error_mask = mask_list[worker][count].eq(0).float()
            error_p.data = p.data * error_mask
    return None

def benchmark_sparse(top_k, models,num_w, new_model, device, errors):
    initialize_zero(new_model)
    for worker in range(num_w):
        grad_flattened = torch.empty(0).to(device)
        for p in models[worker].parameters():
            a = p.data.flatten().to(device)
            grad_flattened = torch.cat((a, grad_flattened), 0).to(device)
        top_grads = torch.topk(grad_flattened.abs(),top_k)[0]
        min_grad = top_grads.min().to(device)
        for p,new_p,error_p in zip(models[worker].parameters(),new_model.parameters(),errors[worker].parameters()):
            sparse_mask = (p.data.abs().ge(min_grad)).float()
            print(sparse_mask.mul(1/num_w))
            new_p.data.add_(p.data * sparse_mask.mul(1/num_w))
            error_mask = 1-sparse_mask
            error_p.data = p.data * error_mask

