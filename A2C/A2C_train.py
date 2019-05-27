#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym, os
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Multinomial
import torchvision.transforms as T
from datetime import datetime
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecFrameStack
from model import ActorCritic
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 123
num_envs = 16
num_frames = 4

def custom_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0,
                     allow_early_resets=True):
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(
                logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)
            return wrap_deepmind(env, **wrapper_kwargs)
        
        return _thunk
    
    set_global_seeds(seed)
    return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])


env = custom_atari_env('BreakoutNoFrameskip-v4', num_env=num_envs, seed=42)
env = make_atari_env('PongNoFrameskip-v0', num_env=num_envs, seed=42)
env = VecFrameStack(env, n_stack=num_frames)


# Helper function

# In[3]:



def process_rollout(gamma = 0.99, lambd = 1.0):
    _, _, _, _, last_values = steps[-1]
    returns = last_values.data

    advantages = torch.zeros(num_envs, 1)
    #if cuda: advantages = advantages.cuda()

    out = [None] * (len(steps) - 1)

    # run Generalized Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        rewards, masks, actions, policies, values = steps[t]
        _, _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * gamma * masks

        deltas = rewards + next_values.data * gamma * masks - values.data
        advantages = advantages * gamma * lambd * masks + deltas

        out[t] = actions, policies, values, returns, advantages

    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))




learn_rate = 1e-4
val_coeff = 0.5
ent_coeff = 0.01
rollout_steps = 50
max_steps = 5e7

# Get number of actions from gym action space
n_actions = env.action_space.n

actor_critic = ActorCritic(n_actions).to(device)

optimizer = optim.Adam(actor_critic.parameters(), lr=learn_rate)

scores_all_envs = []
total_steps_list = []
envs_list = []

steps = []
ep_rewards = [0.] * num_envs
total_steps = 0

state = env.reset()
while total_steps < max_steps:
    for _ in range(rollout_steps):
        state = torch.from_numpy(state.transpose((0, 3, 1, 2))).float() / 255.
        logit, values = actor_critic(state)
        
        probs = F.softmax(logit)
        actions = probs.multinomial(1).data
        
        state, rewards, dones, _ = env.step(actions.cpu().numpy())


        total_steps += num_envs
        for i, done in enumerate(dones):
                ep_rewards[i] += rewards[i]
                if done:
                    #print(dones)
                    scores_all_envs.append(ep_rewards[i])
                    total_steps_list.append(total_steps)
                    envs_list.append(i)
                    if i == 0:
                        print('Timestamp: {}, Steps: {}, Score env: {}, Env no.: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), total_steps, ep_rewards[i], i+1))
                    ep_rewards[i] = 0
                    #print(ep_rewards)
        masks = (1. - torch.from_numpy(np.array(dones, dtype=np.float32))).unsqueeze(1)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1)
        steps.append((rewards, masks, actions, logit, values))

    final_state = torch.from_numpy(state.transpose((0, 3, 1, 2))).float() / 255.
    _, final_values = actor_critic(final_state)
    steps.append((None, None, None, None, final_values))
    actions, logit, values, returns, advantages = process_rollout()
    
    probs = F.softmax(logit)
    log_probs = F.log_softmax(logit)
    log_action_probs = log_probs.gather(-1, actions)

    policy_loss = (-log_action_probs * advantages).sum()
    value_loss = (0.5*(values - returns) ** 2.).sum()
    entropy_loss = (log_probs * probs).sum()

    ac_loss = (policy_loss + value_loss * val_coeff + entropy_loss * ent_coeff)

    ac_loss.backward()
    nn.utils.clip_grad_norm(actor_critic.parameters(), 50.)
    optimizer.step()
    optimizer.zero_grad()
    steps = []
    


# In[5]:


env.close()
torch.save(actor_critic, 'results/actor_critic.pkl')
plt.plot(scores_all_envs)






