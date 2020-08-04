import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='mnist', help='mnist, fmnist, cifar10')
    parser.add_argument('--nn_name', type=str, default='mnist', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--dataset_dist', type=str, default='iid', help='distribution of dataset; iid or non_iid')

    # Federated params
    parser.add_argument('--num_client', type=int, default=4, help='number of clients')
    parser.add_argument('--bs', type=int, default=64, help='batchsize')
    parser.add_argument('--num_epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning_rate')
    parser.add_argument('--sparsity_window', type=int, default=100, help='largest grad entry is chosen within this window')
    parser.add_argument('--lr_change', type=list, default=[100, 225], help='determines the at which epochs lr will decrease')
    parser.add_argument('--errorDecay',type=bool,default=False, help='error correction decays over the time.')
    parser.add_argument('--errDecayVals',type=list,default=[100,0.9], help='list[0] gives at which epoch errorCorr will multipled with l[1]')
    parser.add_argument('--warmUp', type=bool, default=False, help='LR warm up.')

    # MajorityVoting Params
    parser.add_argument('--majority_Sparsity', type=int, default=100, help='Determines the layer majority, -1 for disable')
    parser.add_argument('--layerMajority', type=int, default=-1, help='Determines the layer majority, -1 for disable')
    parser.add_argument('--K_val', type=int, default=1000, help='Determines the layer majority, -1 for disable')
    parser.add_argument('--RandomSelect', type=int, default=-1, help='Randomly select percent of worker majority, -1 for disable')
    parser.add_argument('--RandomWorkers', type=list, default=[0,4],help='Randomly select percent of worker majority, -1 for disable')
    parser.add_argument('--type', type=int, default=0, help='0 for MajorityVoting, 1 for Weighted-MV')
    parser.add_argument('--m_type', type=int, default=0, help='0 for normal, 1 for special momentum')
    parser.add_argument('--momentum_val', type=float, default=0.8, help='momentum value')
    parser.add_argument('--w_custom_mask', type=int, default=0, help='0 for majorityMask, 1 for custom Majority Mask')

    # Quantization params
    parser.add_argument('--quantization', type=bool, default=False,help='apply quantization or not')
    parser.add_argument('--num_groups', type=int, default=16, help='Number Of groups')
    parser.add_argument('--denominator', type=float, default=1.2, help='divide groups by this')
    args = parser.parse_args()
    return args