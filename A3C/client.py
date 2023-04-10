import argparse

import sys
sys.path.append("c:/users/ningc/desktop/TPDS 1.0_code")

import os
import torch
import numpy as np
import torch.nn.functional as F
import A3C.envs as envs
from A3C.model import ActorCritic
#from A3C.videoPlayer import videoPlayer

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def start_client(ImageBuffer,MotionVectorBuffer):
    os.environ['OMP_NUM_THREADS'] = '1'
    print("loading DRL model...")
    env=envs.env(ImageBuffer,MotionVectorBuffer)
    state=env.reset()
    N_S,A_S=env.StateActiondim()
    model = ActorCritic(A_S)
    #model.load_state_dict(torch.load("A3C.weights"))
    print("finished!")
    done=False

    while not done:
        print("\033[1;31mstate:",state,"\033[0m ")
        # logits, _=model(v_wrap(state[None, : ]))
        # prob = F.softmax(logits, dim=-1)
        #action = prob.multinomial(num_samples=1).detach()
        action=30*5-1
        #state,done=env.step(action.numpy()[0][0])
        state,done=env.step(action)
        print("current state:",state)
    print("detection finished!")



