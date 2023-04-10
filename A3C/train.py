import torch
import torch.nn.functional as F
import torch.optim as optim
import A3C.envs as envs
from A3C.model import ActorCritic
import A3C.constant as constant
import numpy as np

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def to_tensor(state,dtype=np.float32):
    tensor_state = [torch.from_numpy(state[0][None, None, :]), torch.from_numpy(state[1][None, None, :]),
                    torch.from_numpy(np.array([state[2], state[3]], np.float32)[None, :])]
    return tensor_state

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def train(rank, args, shared_model, counter, lock, optimizer=None):

    env = envs.TrainEnv()
    _, A_S = env.StateActiondim()
    model = ActorCritic(A_S, constant.k)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()
    state, image_state = env.reset()
    done = True
    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            logits, value = model(to_tensor(state), image_state)
            prob = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)
            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)
            state, image_state, reward, done, _, _, _ = env.step(action.numpy()[0][0])
            # if done:
            #     reward=100000
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)
            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state, image_state = env.reset()

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            _, value = model(to_tensor(state), image_state)
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
