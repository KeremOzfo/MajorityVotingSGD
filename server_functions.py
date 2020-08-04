import torch
import math
import time
import numpy as np

def pull_model(model_user, model_server):

    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_user.data = param_server.data[:] + 0

    return None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def zero_grad_ps(model):
    for param in model.parameters():
        param.grad = torch.zeros_like(param.data)

    return None


def push_grad(model_user, model_server, num_cl):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_server.grad.data += param_user.grad.data / num_cl
    return None

def push_model(model_user, model_server, num_cl):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_server.data += param_user.data / num_cl
    return None

def initialize_zero(model):
    for param in model.parameters():
        param.data.mul_(0)
    return None


def update_model(model, lr):
    for param in model.parameters():
        param.data.add_(-lr, param.grad.data)
    return None


def get_grad_flattened(model, device):
    grad_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        if p.requires_grad:
            a = p.grad.data.flatten().to(device)
            grad_flattened = torch.cat((grad_flattened, a), 0)
    return grad_flattened

def get_model_flattened(model, device):
    model_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        a = p.data.flatten().to(device)
        model_flattened = torch.cat((model_flattened, a), 0)
    return model_flattened

def get_model_sizes(model):
    # get the size of the layers and number of eleents in each layer.
    # only layers that are trainable
    net_sizes = []
    net_nelements = []
    for p in model.parameters():
        if p.requires_grad:
            net_sizes.append(p.data.size())
            net_nelements.append(p.nelement())
    return net_sizes, net_nelements



def unshuffle(shuffled_vec, seed):
    orj_vec = torch.empty(shuffled_vec.size())
    perm_inds = torch.tensor([i for i in range(shuffled_vec.nelement())])
    perm_inds_shuffled = shuffle_deterministic(perm_inds, seed)
    for i in range(shuffled_vec.nelement()):
        orj_vec[perm_inds_shuffled[i]] = shuffled_vec[i]
    return orj_vec


def shuffle_deterministic(grad_flat, seed):
  # Shuffle the list ls using the seed `seed`
  torch.manual_seed(seed)
  idx = torch.randperm(grad_flat.nelement())
  return grad_flat.view(-1)[idx].view(grad_flat.size())


def get_indices(net_sizes, net_nelements):
    # for reconstructing grad from flattened grad
    ind_pairs = []
    ind_start = 0
    ind_end = 0
    for i in range(len(net_sizes)):

        for j in range(i + 1):
            ind_end += net_nelements[j]
        # print(ind_start, ind_end)
        ind_pairs.append((ind_start, ind_end))
        ind_start = ind_end + 0
        ind_end = 0
    return ind_pairs


def make_grad_unflattened(model, grad_flattened, net_sizes, ind_pairs):
    # unflattens the grad_flattened into the model.grad
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            temp = grad_flattened[ind_pairs[i][0]:ind_pairs[i][1]]
            p.grad.data = temp.reshape(net_sizes[i])
            i += 1
    return None

def make_model_unflattened(model, model_flattened, net_sizes, ind_pairs):
    # unflattens the grad_flattened into the model.grad
    i = 0
    for p in model.parameters():
        temp = model_flattened[ind_pairs[i][0]:ind_pairs[i][1]]
        p.data = temp.reshape(net_sizes[i])
        i += 1
    return None


def make_sparse_grad(grad_flat, sparsity_window,device):
    # sparsify using block model
    num_window = math.ceil(grad_flat.nelement() / sparsity_window)

    for i in range(num_window):
        ind_start = i * sparsity_window
        ind_end = min((i + 1) * sparsity_window, grad_flat.nelement())
        a = grad_flat[ind_start: ind_end]
        ind = torch.topk(a.abs(), k=1, dim=0)[1] #return index of top not value
        val = a[ind]
        ind_true = ind_start + ind
        grad_flat[ind_start: ind_end] *= torch.zeros(a.nelement()).to(device)
        grad_flat[ind_true] = val

    return None

def adjust_learning_rate(optimizer, epoch,lr_change, lr):

    lr_change = np.asarray(lr_change)
    loc = np.where(lr_change == epoch)[0][0] +1
    lr *= (0.1**loc)
    lr = round(lr,3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def lr_warm_up(optimizers, num_workers, epoch,start_lr):
    for cl in range(num_workers):
        for param_group in optimizers[cl].param_groups:
            if epoch == 0:
                param_group['lr'] = 0.1
            else:
                lr_change = (start_lr - 0.1) / 4
                param_group['lr'] = (lr_change * epoch) + 0.1


def get_LR(optimizer):
    lr =None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr



def groups(grad_flat, group_len,denominator,device):
    sparseCount = torch.sum(grad_flat!=0)
    sparseCount= sparseCount.__int__()
    vals, ind = torch.topk(grad_flat.abs(),k=sparseCount, dim=0)
    group_boundries = torch.zeros(group_len + 1).to(device)
    group_boundries[0] = vals[0].float()
    sign_mask = torch.sign(grad_flat[ind])
    for i in range(1,group_len):
        group_boundries[i] = group_boundries[i-1] /denominator
    startPoint =0
    newVals = torch.zeros_like(vals)
    startPointz =[]
    for i in range(group_len):
        if vals[startPoint] > group_boundries[i+1]:
            startPointz.append(startPoint)
            for index,val in enumerate(vals[startPoint:vals.numel()]):
                if val <= group_boundries[i+1] and group_boundries[i+1] !=0:
                    newVals[startPoint:startPoint+index] = torch.mean(vals[startPoint:startPoint+index])
                    startPoint += index
                    break
                elif group_boundries[i+1]==0:
                    newVals[startPoint:vals.numel()] = torch.mean(vals[startPoint:vals.numel()])
                    break
    newVals *= sign_mask
    grad_flat *= 0
    grad_flat[ind] = newVals



def collectMajority_grad(grad_flat,majority_pool,layer_majority, total_majorty,ind_pairs,random,mode,device):
    clone_grad = torch.clone(grad_flat).to(device)
    majority_len = math.ceil(grad_flat.numel() / total_majorty)
    inds = torch.empty(0).to(device)
    vals = torch.empty(0).to(device)
    if layer_majority > 0:
        for layer in ind_pairs:
            startPoint = (layer[0])
            endPoint = (layer[1])
            layer_len = endPoint - startPoint
            l_top_k = math.ceil(layer_len / layer_majority)
            l_val, l_ind = torch.topk((grad_flat[startPoint:endPoint]).abs(), k=l_top_k, dim=0)
            l_ind.add_(startPoint)
            inds = torch.cat((inds.float(), l_ind.float()), 0)
            vals = torch.cat((vals.float(),l_val.float()), 0)
        inds = inds.long()
        clone_grad[inds] = 0
    inds = inds.long()
    if inds.numel() < majority_len:
        topk = majority_len - inds.numel()
        vals_, inds_ = torch.topk(clone_grad.abs(), k=topk, dim=0)
        inds = torch.cat((inds,inds_), 0)
        vals = torch.cat((vals, vals_), 0)
    clone_grad *= 0
    clone_grad[inds] = vals
    vals, inds = torch.topk(clone_grad.abs(), k=majority_len, dim=0) ## reorder vals and inds
    if random >0:
        random_select = math.ceil(inds.numel()/random)
        inds = inds.cpu().numpy()
        inds = np.random.choice(inds,random_select) ## randomly select
        inds = torch.tensor(inds)
        inds = inds.to(device) # send to cuda again
        vals = clone_grad[inds] # get corresponding values
    majority_pool[inds] += 1 ## make normal Majority-V for all ind Values
    if mode !=0:
        for i in range(3):
            gValue = vals[0] / ((3-i)**2) # make Weighted Majority-V starting from lowest group bound
            mask = vals.ge(gValue)
            majority_pool[torch.masked_select(inds,mask)] += 1
    return None

def worker_voting(prev_mask, worker_grad, sparsity, k_val, device):
    sparse_val = math.ceil(worker_grad.numel() / sparsity)
    k_sparse = math.ceil(worker_grad.numel() / k_val)
    w_val, w_inds = torch.topk(worker_grad.abs(), k = sparse_val, dim=0)
    worker_top = torch.zeros_like(worker_grad).to(device)
    worker_top[w_inds] = 1
    m_drop = worker_top.mul(prev_mask)
    #print(torch.sum(m_drop), torch.sum(prev_mask))
    m_drop = prev_mask.sub(1, m_drop)
    #print(torch.sum(m_drop))
    m_Atarget = worker_top.sub(1, worker_top.mul(prev_mask))
    m_dropCount = torch.sum(m_drop).__int__()
    if m_dropCount < k_sparse:
        k_sparse = m_dropCount
    m_dropTarget = worker_grad.mul(m_drop)
    drop_val, drop_ind = torch.topk(m_dropTarget.abs(), k=sparse_val, dim=0)
    drop_mask = drop_val > 0
    drop_ind = torch.flip(torch.masked_select(drop_ind, drop_mask), dims=[0])
    add_val , add_ind = torch.topk((worker_grad.mul(m_Atarget)).abs(), k=k_sparse, dim=0)
    return add_ind, drop_ind[0:k_sparse]