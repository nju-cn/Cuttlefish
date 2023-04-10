'''
The entrance of training actor-critic model
'''
import sys
sys.path.append("c:/users/ningc/desktop/TPDS 1.0_code")
import argparse
import os
import torch
import time
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import A3C.constant as constant
import A3C.my_optim as my_optim
import A3C.envs as envs
from A3C.model import ActorCritic
from A3C.test import test
from A3C.train import train

# Parameters of training process
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=2,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=10,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    env = envs.TrainEnv()
    N_S, A_S = env.StateActiondim()
    shared_model = ActorCritic(A_S, constant.k)
    shared_model.share_memory()
    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter, True))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
