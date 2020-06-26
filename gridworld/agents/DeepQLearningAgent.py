import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# defining an agent who utilises DEEP Q-learning
# rather than utilise a Q-table to store all state-reward pairs
# uses a neural network to learn a distribution
# takes state as input, generates Q-value for all possible actions as output
class DeepQLearningAgent():
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=1):
        # % chance that the agent will perform a random exploratory action
        self.epsilon = epsilon
        # learning rate -> how much the difference between new values is changed
        self.alpha = alpha
        # discount factor -> used to balance future/immediate reward
        self.gamma = gamma
        # optimiser adjusts the weights of the neural network to minimise loss
        self.optimiser = torch.optim.Adam()


    def choose_action(self, available_actions):
        """Returns the optimal action for the state from the Qq value as predicted by the neural network"""
        if np.random.uniform(0,1) < self.epsilon:
            # chooses a random exploratory action if chance is under epsilon
            action = np.random.choice(available_actions)
        else:
            # gets the actions associated with that state
            q_values_of_state = 0
            # finds the maximum possible reward
            max_value = max(q_values_of_state.values())
            # chooses an action with the maximum possible reward
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == max_value])
        return action

    def learn(self, old_state, reward, new_state, action):
        return "lol"
