import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

import argparse

import gym

seed = 1
rank = 16

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--env-name',type=str, default = 'PongDeterministic-v4',
                    help = 'default environment is PongDeterminisitc-v4')
parser.add_argument('--model', type = str, default = 'actor.pkl',
                    help = 'default model to load is actor.pkl')

args = parser.parse_args()


env_name = args.env_name

torch.manual_seed(seed + rank)

env = create_atari_env(env_name)
env.seed(seed + rank)

#env = gym.make("Pong-v0")
#env = make_env(env)

model = ActorCritic(env.observation_space.shape[0], env.action_space)
testModel = torch.load('C:/Users/Frederik/Documents/GitHub/ReinforcementLearning/pytorch-a3c-master/results/' + args.model)
#model.load_state_dict(testModel.state_dict())

model.eval()

state = env.reset()
state = torch.from_numpy(state)
reward_sum = 0
done = True

start_time = time.time()

n_iter = 2
n = 0
max_episode_length = 100000

    # a quick hack to prevent the agent from stucking
actions = deque(maxlen=100)
episode_length = 0
while n < n_iter:
        env.render()
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(testModel.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            n+= 1
            time.sleep(5)

        state = torch.from_numpy(state)
env.close()


