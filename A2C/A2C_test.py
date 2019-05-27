import gym, os
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecFrameStack
from model import ActorCritic
from gym import wrappers


import matplotlib.pyplot as plt




seed = 123
num_envs = 16
num_frames = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_iter = 4

env = make_atari_env('PongNoFrameskip-v0', num_env=1, seed=24)
#env = wrappers.Monitor(env, "results/A2C-results", force=True)
env = VecFrameStack(env, n_stack=num_frames)

n_actions = env.action_space.n

actor_critic = ActorCritic(n_actions).to(device)
testModel = torch.load('results/actor_critic.pkl')
actor_critic.load_state_dict(testModel.state_dict())

state = env.reset()
for _ in range(n_iter):
    for _ in count():
        env.render()
        state = torch.from_numpy(state.transpose((0, 3, 1, 2))).float() / 255.
        logit, values = actor_critic(state)

        probs = F.softmax(logit)
        #actions = probs.multinomial(1).data
        actions = torch.argmax(probs, dim=1)
        state, rewards, dones, _ = env.step(actions.cpu().numpy())
        if dones:
            break
env.close()