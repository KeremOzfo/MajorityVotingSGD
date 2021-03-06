from train_funcs import *
import numpy as np
from parameters import *
import torch
import random
import datetime
import os

device = torch.device("cpu")
args = args_parser()

if __name__ == '__main__':
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    simulation_ID = int(random.uniform(1,999))
    args = args_parser()
    for arg in vars(args):
       print(arg, ':', getattr(args, arg))
    results = []
    x = datetime.datetime.now()
    date = x.strftime('%b') + '-' + str(x.day)
    newFile = date + '_{}_{}'.format(simulation_ID,args.MajorityType)
    if not os.path.exists(os.getcwd() + '/Results'):
        os.mkdir(os.getcwd() + '/Results')
    n_path = os.path.join(os.getcwd(), 'Results', newFile)
    for i in range(7):
        if args.MajorityType =='vanilla':
            accs = train_vanilla(args, device)
        elif args.MajorityType =='add_drop':
            accs = train_add_drop(args, device)
        elif args.MajorityType == 'random':
            accs = train_random(args, device)
        else:
            raise Exception('Such Majority Type is not available')
        if i == 0:
            os.mkdir(n_path)
            f = open(n_path + '/simulation_Details.txt', 'w+')
            f.write('simID = ' + str(simulation_ID) + '\n')
            f.write('############## Args ###############' + '\n')
            for arg in vars(args):
                line = str(arg) + ' : ' + str(getattr(args, arg))
                f.write(line + '\n')
            f.write('############ Results ###############' + '\n')
            f.close()
        s_loc = date+'_MajorityVoting_{}_Q_{}_{}'.format(args.MajorityType,args.quantization,i)
        s_loc = os.path.join(n_path,s_loc)
        np.save(s_loc,accs)
        f = open(n_path + '/simulation_Details.txt', 'a+')
        f.write('Trial ' + str(i) + ' results at ' + str(accs[args.num_epoch]) + '\n')
        f.close()