#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:17:46 2019

@author: a.k.salk
"""

import numpy as np
import matplotlib.pyplot as plt

num_bandits = 10
steps = 10000
alpha = 0.1
epsilon = 0.1
mean=0
std=0.01

# Random walk paramters
plus=0.01
minus=-plus

num_actions = 1

actions = []

# Creating bandits
bandits = []
for i in range(num_bandits):
    bandits.append([np.random.normal(mean, std), std])

# h loop defines the number of re-runs to generate a more fluent distribution on the problem. 
for h in range(100):
    opt_a = []
    
    # Collecting max value for and position
    m_bandit = 0
    b_bandit = 0
    for i in range(num_bandits):
        if bandits[i][0] > m_bandit:
            m_bandit = bandits[i][0]
            b_bandit = i
            
    q = np.zeros(num_bandits)
    n = np.zeros(num_bandits)

    for i in range(steps):
        
        # Creating the random walk
        for bandit in bandits:
            if np.random.uniform() < 0.5:
                bandit[0] += np.random.normal(mean, std)
            else:
                bandit[0] -= np.random.normal(mean, std)
        selected_bandit = np.argmax(q)

        if np.random.uniform() < epsilon:
            selected_bandit = np.random.choice(len(q))

        reward = np.random.normal(bandits[selected_bandit][0], bandits[selected_bandit][1])

        n[selected_bandit] += 1
        q[selected_bandit] += alpha * (reward - q[selected_bandit])

        opt_a.append(int(selected_bandit == b_bandit))

        if i <= num_actions:
            action = sum(opt_a)/len(opt_a)
        else:
            action = sum(opt_a[-num_actions:])/num_actions

        if h == 0:
            actions.append(action)
        else:
            actions[i] += (action - actions[i]) / (h + 1)

plt.axis([0, steps, -1, 1])
plt.plot(range(steps), actions)
plt.show()