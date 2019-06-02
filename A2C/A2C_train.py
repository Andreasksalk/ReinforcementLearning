#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
# Stable baselines is a further development of openAI baselines
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

def custom_dummy_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0,
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

# Creating a dummy vec env since it allows to bypass an error in SubprocVecEnv from baselines
env = custom_dummy_env('PongNoFrameskip-v0', num_env=num_envs, seed=42)
env = make_atari_env('PongNoFrameskip-v0', num_env=num_envs, seed=42)
env = VecFrameStack(env, n_stack=num_frames)


def controller(gamma = 0.99):
    _, _, _, _, last_values = steps[-1]
    returns = last_values.data


    # Create advantage tensor with the length of virtual environments.
    advantage = torch.zeros(num_envs, 1)

    out = [None] * (len(steps) - 1)

    # Takes the bootstrapped values from all the environments and calculates adv.
    for t in reversed(range(len(steps) - 1)):
        reward, mask, action, logit, value = steps[t]
        _, _, _, _, next_values = steps[t + 1]

        returns = reward + returns * gamma * mask

        delta = reward + next_values.data * gamma * mask - value.data
        advantage = advantage * gamma * mask + delta

        out[t] = actions, logit, values, returns, advantage

    # return data as batched Tensors
    return map(lambda x: torch.cat(x, 0), zip(*out))




learn_rate = 1e-4
val_coeff = 0.5
ent_coeff = 0.01
bootstrap_steps = 50
max_steps = 5e7

# Get number of actions from gym action space
n_actions = env.action_space.n

actor_critic = ActorCritic(n_actions).to(device)

optimizer = optim.Adam(actor_critic.parameters(), lr=learn_rate)

scores_all_envs = []
total_steps_list = []
envs_list = []

steps = []
ep_rewards = [0] * num_envs
total_steps = 0

state = env.reset()
while total_steps < max_steps:
    for _ in range(bootstrap_steps):
        # Reshaping tensor to have (batch, frames, width, height), and normalizing color values
        state = torch.from_numpy(state.transpose((0, 3, 1, 2))).float() / 255.
        logit, values = actor_critic(state)

        # Get the probabilites of taking an action in a given state and sample 1 action per env
        probs = F.softmax(logit)
        action = probs.multinomial(1).data
        
        state, reward, boolean, _ = env.step(action.cpu().numpy())


        total_steps += num_envs
        for i, done in enumerate(boolean):
                ep_rewards[i] += reward[i]
                if done:
                    # if an environments is done, append its values
                    scores_all_envs.append(ep_rewards[i])
                    total_steps_list.append(total_steps)
                    envs_list.append(i)
                    # Print first environment as tester
                    if i == 0:
                        print('Timestamp: {}, Steps: {}, Score env: {}, Env no.: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), total_steps, ep_rewards[i], i+1))
                    ep_rewards[i] = 0
        mask = (1. - torch.from_numpy(np.array(dones, dtype=np.float32))).unsqueeze(1)
        reward = torch.from_numpy(reward).float().unsqueeze(1)
        steps.append((reward, mask, action, logit, values))

    final_state = torch.from_numpy(state.transpose((0, 3, 1, 2))).float() / 255.
    _, final_value = actor_critic(final_state)
    steps.append((None, None, None, None, final_value))
    action, logit, value, returns, advantage = controller()
    
    probs = F.softmax(logit)
    log_probs = F.log_softmax(logit)
    log_action_probs = log_probs.gather(-1, action)

    # Calculate loss from the probs and advantage
    actor_loss = (-log_action_probs * advantage).sum()
    critic_loss = (0.5*(value - returns) ** 2).sum()
    entropy_loss = (log_probs * probs).sum()


    # Increase entropy loss to encourage exploration
    ac_loss = (actor_loss + critic_loss * val_coeff + entropy_loss * ent_coeff)

    ac_loss.backward()
    #nn.utils.clip_grad_norm(actor_critic.parameters(), 50)
    optimizer.step()
    optimizer.zero_grad()
    steps = []

env.close()
torch.save(actor_critic, 'results/actor_critic.pkl')
plt.plot(scores_all_envs)






