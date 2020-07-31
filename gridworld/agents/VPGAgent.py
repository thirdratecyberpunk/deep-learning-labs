# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:09:24 2020

@author: Lewis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import sys

class PolicyNetwork(nn.Module):
    """
    Neural network that attempts to learn underlying policy mapping
    states to actions
    """
    def __init__(self, in_features=48, num_actions=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x))
        return x

def vector_to_tensor(vector):
    return torch.from_numpy(vector).float()

# defining an agent who utilises Vanilla Policy Gradient
# agent attempts to directly learn the policy function mapping S to A
# need to therefore directly optimise the policy function
# can learn determinisitic and stochastic policies
# capable of handling Partially Observable Markov Decision Processes
class VPGAgent():
    # initialises
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=0.5):
        # checking CUDA support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # environment the agent operates in
        self.environment = environment
        # gamma value
        self.gamma = gamma
        # policy network
        self.policy_network = PolicyNetwork(48, 4)
        self.rewards = []
        self.actions = []
        self.states = []
        # optimiser
        self.optimiser = torch.optim.Adam(self.policy_network.parameters())
        
    def choose_action(self, available_actions):
        """Returns the optimal action for the state from the policy"""
        # gets the vector of probabilities from the neural network
        probabilities = self.policy_network(vector_to_tensor(self.environment.agents_on_map_vector()))
        # samples from available action space using probability distribution
        # if the agent has finished an episode, fix gradients
        sampler = Categorical(probabilities)
        action = sampler.sample()
        chosen_action = available_actions[action.item()]        
        return chosen_action

    def learn(self, old_state, reward, new_state, action):
        # add values to collection of episode details
        self.rewards.append(reward)
        self.actions.append(action.value)
        self.states.append((self.environment.coords_to_aom_vector(new_state)))


    def finish_episode(self):
        # calculating the gradient at the end of an episode
        # preprocess rewards
#        R = torch.sum(torch.tensor(self.rewards))
        self.rewards = np.array(self.rewards)
        R = torch.tensor([np.sum(self.rewards[i:]*(self.gamma**np.array(range(i, len(self.rewards))))) for i in range(len(self.rewards))])
        # preprocess states and actions
        states_tensor = torch.tensor(self.states).float().to(self.device)
        actions_tensor = torch.tensor(self.actions).to(self.device)
        
        probs = self.policy_network(states_tensor)
        sampler = Categorical(probs)
        log_probs = -sampler.log_prob(actions_tensor)   # "-" because it was built to work with gradient descent, but we are using gradient ascent
        pseudo_loss = torch.sum(log_probs * R) # loss that when differentiated with autograd gives the gradient of J(θ)
        # update policy weights
        self.optimiser.zero_grad()
        pseudo_loss.backward()
        self.optimiser.step()
        # clear 
        self.rewards = []
        self.actions = []
        self.states = []
