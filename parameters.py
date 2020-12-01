import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=1, help='cuda:No')

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='mnist, fmnist, cifar10')
    parser.add_argument('--nn_name', type=str, default='resnet18', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--dataset_dist', type=str, default='iid', help='distribution of dataset; iid or non_iid')
    parser.add_argument('--numb_cls_usr', type=int, default=3, help='number of class per user if non_iid2 selected')

    # Federated params
    parser.add_argument('--alfa', type=float, default=0.95, help='weigths ')
    parser.add_argument('--num_client', type=int, default=10, help='number of clients')
    parser.add_argument('--num_epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--LocalIter', type=int, default=5, help='communication workers')
    parser.add_argument('--bs', type=int, default=64, help='batchsize')
    parser.add_argument('--lr', type=float, default=0.5, help='learning_rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--W_decay', type=float, default=1e-4, help='weight decay Value')
    parser.add_argument('--lr_change', type=list, default=[150, 225],
                        help='determines the at which epochs lr will decrease')
    parser.add_argument('--warmUp', type=bool, default=True, help='LR warm up.')

    args = parser.parse_args()
    return args
