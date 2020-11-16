# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 11:36:34 2020

@author: Lewis
Testing actor-critic learning on cartpole
"""

import collections
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import tqdm
from itertools import count

from matplotlib import pyplot as plt
from typing import Any, List, Sequence, Tuple

import sys

# setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# creating environment
env = gym.make("CartPole-v0")
# getting information about size of state space 
OBSERVATION_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n
lr = 0.0001

# setting seeds for reproducibility
seed = 0
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# actor agent: agent that learns the policy function
class ActorAgent(nn.Module):
    """ 
    given a state s, returns a probability distribution over the set of 
    possible actions
    """
    def __init__(self, observation_size, action_size):
        super(ActorAgent, self).__init__()
        # environment sizes
        self.observation_size = observation_size
        self.action_size = action_size
        # layers
        self.fc1 = nn.Linear(self.observation_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, self.action_size)
        
    def forward(self, x):
        """
        outputs a probability distribution for each actions in space
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        distribution = Categorical(F.softmax(x, dim=-1))
        return distribution


# critic agent: agent that defines the expected return for an agent
class CriticAgent(nn.Module):
    """
    given an agent's initial state and a policy, return the expected return
    """
    def __init__(self,observation_size,action_size):
        super(CriticAgent, self).__init__()
        # environment sizes
        self.observation_size = observation_size
        self.action_size = action_size
        # layers
        self.fc1 = nn.Linear(self.observation_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)


    def forward(self, x):
        """
        outputs the expected return
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def computeReturns(next_value, rewards, masks, gamma=0.99):
    """
    given a predicted value from the critic agent, update the returns
    """
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns
  
def trainIterations(max_iter=1000, target_average=195):
    # run agent in environment to generate training data
    actor = ActorAgent(OBSERVATION_SPACE, ACTION_SPACE)
    critic = CriticAgent(OBSERVATION_SPACE, ACTION_SPACE)

    # setting up optimisers
    actor_optimiser = torch.optim.SGD(actor.parameters(), lr=0.001)
    critic_optimiser = torch.optim.SGD(critic.parameters(), lr=0.001)

    # repeating the training process until 
    # success criterion or max episodes has been reached
    for iter in range(max_iter):
        # storing round information
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()
        
        # running the simulation until it is done
        for i in count():
            env.render()
            state = torch.FloatTensor(state).to(device)
            # get the probability distribution and expected return
            distribution, value = actor(state), critic(state)
            # choose an action via sampling from distribution
            action = distribution.sample()
            # takes action in environment
            next_state, reward, done, _ = env.step(action.to(device).numpy())
            
            # records log of prob of chosen action
            log_prob = distribution.log_prob(action).unsqueeze(0)
            # add to entropy: i.e. representing how much uncertainty is in
            # the probability distribution
            entropy += distribution.entropy().mean()
            # add information about current state to round information
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            
            state = next_state
            
            # if agent has reached the terminal state of the environment
            if done:
                print(f"Iteration {iter}, score {i}")
                break
         
        # calculating returns for that episode
        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = computeReturns(next_value, rewards, masks)
        
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        
        # advantage represents how much better an action is given a particular
        # state over a random action for that state
        # "how much better is this action than others on average?"
        
        advantage = returns - values
        
        # calculate the loss for each agent
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        # update models
        
        actor_optimiser.zero_grad()
        critic_optimiser.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        actor_optimiser.step()
        critic_optimiser.step()
            
        # if the agent has "solved" the problem by reaching the target average
        if (sum(rewards) / len(rewards) >= target_average):
            print(f"Problem solved at average of {target_average} on iteration {iter}")
            env.close()
            sys.exit(1)
    env.close()
    
if __name__ == '__main__':
    trainIterations()