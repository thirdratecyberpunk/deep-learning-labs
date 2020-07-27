import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# stores a sequence of state transitions (i.e state, action, next state and the reward)
# this is used to allow reuse of already observed transitions
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    # TODO: make in_features not hardcoded based on environment size
    # could add a method to get state size in environments?
    def __init__(self, in_features=48, num_actions=4):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def vector_to_tensor(vector):
    return torch.from_numpy(vector).float()

# defining an agent who utilises DEEP Q-learning
# rather than utilise a Q-table to store all state-reward pairs
# uses a neural network to learn a distribution
# takes state as input, generates Q-value for all possible actions as output
class DeepQLearningAgent():
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=1, batch_size=128):
        # checking CUDA support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print (self.device)
        # environment that agent is attempting to explore
        self.environment = environment
        # % chance that the agent will perform a random exploratory action
        self.epsilon = epsilon
        # learning rate -> how much the difference between new values is changed
        self.alpha = alpha
        # discount factor -> used to balance future/immediate reward
        self.gamma = gamma
        # neural network outputs Q(state, action) for all possible actions
        # for gridworld, all actions are UP, DOWN, LEFT, RIGHT
        # therefore output should contain 4 outputs
        self.policy_qnn = DQN()
        # neural network that acts as the Q function approximator
        self.target_qnn = DQN()
        # loss function
        self.loss_function = nn.MSELoss()
        # optimiser
        self.optimiser = optim.SGD(self.policy_qnn.parameters(),lr=0.001,momentum=0.9)
        # memory containing state transition encountered by the agent
        self.memory = ReplayMemory(1000)
        # size of a batch of replays to sample from
        self.batch_size = batch_size

    def choose_action(self, available_actions):
        """Returns the optimal action for the state from the Q value as predicted by the neural network"""
        if np.random.uniform(0,1) < self.epsilon:
            # chooses a random exploratory action if chance is under epsilon
            action = np.random.choice(available_actions)
        else:
            # gets the values associated with action states from the neural network
            q_values = self.policy_qnn.forward(vector_to_tensor(self.environment.agents_on_map_vector()))
            q_values_for_states = dict(zip(available_actions, (x.item() for x in q_values)))
            # chooses the action with the best known 
            action = sorted(q_values_for_states.items(), key=lambda x: x[1])[0][0]
        return action

    def learn(self, old_state, reward, new_state, action):
        # add the state transition to the memory
        old_state_as_vector = vector_to_tensor(self.environment.coords_to_aom_vector(old_state)).to(self.device)
        new_state_as_vector = vector_to_tensor(self.environment.coords_to_aom_vector(new_state)).to(self.device)
        action_as_tensor = torch.tensor(action.value).to(self.device)
        reward_as_tensor = torch.tensor(reward).to(self.device)
        self.memory.push(old_state_as_vector, action_as_tensor, new_state_as_vector, reward_as_tensor)
        # if the memory isn't full, skip learning
        if len(self.memory) < self.batch_size:
            return
        # if the memory IS full, sample from states
        transitions = self.memory.sample(self.batch_size)
        # prepare batch of transitions to learn from
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat([torch.unsqueeze(i, 0) for i in batch.action]).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)
        # compute Q(s_t, a)
        # first, calculate the Q value for a given state via model
        old_state_q_values = self.policy_qnn(state_batch)
        # then, select the actions taken according to the policy net
        # TODO: fix this, I think this is kinda hacky?
        old_state_q_values = old_state_q_values.index_select(1, action_batch)
        # hacky bit to just get the first column from the old state q values
        old_state_q_values = old_state_q_values[:,0]
        # compute V(s_{t+1} for NEXT states)
        updated_state_q_values = self.target_qnn(state_batch).max(1)[0].detach()
        expected_state_action_values = updated_state_q_values * self.gamma + reward_batch
        # compute the loss based on difference between actual and expected values
        loss = F.smooth_l1_loss(old_state_q_values, expected_state_action_values.unsqueeze(1))
        # optimise the model
        self.optimiser.zero_grad()
        loss = self.loss_function(old_state_q_values, updated_state_q_values)
#        loss = F.smooth_l1_loss(old_state_q_values, updated_state_q_values)
        loss.backward()
        self.optimiser.step()
        
    def update(self):
        """
        Updates the weights used by the target QNN to use policy QNN weights
        """
        self.target_qnn.load_state_dict(self.policy_qnn.state_dict())

