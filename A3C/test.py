import time
from collections import deque
import torch
import torch.nn.functional as F
import numpy as np
import A3C.envs as envs
import A3C.constant as constant
from A3C.model import ActorCritic


def to_tensor(state, dtype=np.float32):
    tensor_state = [torch.from_numpy(state[0][None, None, :]), torch.from_numpy(state[1][None, None, :]),
                    torch.from_numpy(np.array([state[2], state[3]], np.float32)[None, :])]
    return tensor_state


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def test(rank, args, shared_model, counter, save=False):
    total_reward = 0
    env = envs.TrainEnv(test=True)
    _, A_S = env.StateActiondim()
    model = ActorCritic(A_S, constant.k)
    model.eval()
    state, image_state = env.reset()
    reward_sum = 0
    entropy_sum = 0
    done = True
    start_time = time.time()
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
        with torch.no_grad():
            logits, value = model(to_tensor(state), image_state)
            prob = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            action = prob.multinomial(num_samples=1).detach()
            state, image_state, reward, done, _, _, _ = env.step(action.numpy()[0][0])
            reward = max(min(reward, 1), -1)
            # print(action)
            # state,image_state, reward, done= env.step(prob[0].argmax())
            # state,img_state, reward, done= env.step(np.random.randint(0,105))
            total_reward += reward
            # if reward>0:
            #     countp+=1
            # elif reward<0:
            #     countn+=1
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward
        entropy_sum += entropy[0][0]

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}, entropy {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length, entropy_sum / episode_length))
            if save and counter.value >= 10000:
                torch.save(shared_model.state_dict(), "A3C.weights")
            reward_sum = 0
            entropy_sum = 0
            episode_length = 0
            actions.clear()
            state, image_state = env.reset()

            #print(countp, countn, total_reward)
            random_reward = 0
