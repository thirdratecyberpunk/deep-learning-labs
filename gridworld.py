# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:09:17 2020

@author: Lewis
Implementing some reinforcement learning algorithms in Gridworld as defined by 
https://github.com/michaeltinsley/Gridworld-with-Q-Learning-Reinforcement-Learning-/blob/master/Gridworld.ipynb
and LineWalking
"""

import numpy as np
import matplotlib.pyplot as plt

# defining LineWalking
class LineWalking:
    def __init__(self, width = 10):
        self.height = 1
        self.width = width
        self.cliff = np.zeros((self.height, self.width)) - 1
        
        self.current_location = (0,1)
        self.cliff_edge = (0,0)
        self.end_of_cliff = (0, self.width - 1)
        self.terminal_states = [self.cliff_edge, self.end_of_cliff]
        
        self.cliff[self.cliff_edge] = -100
        self.cliff[self.end_of_cliff] = 100
        
        self.actions = ['LEFT', 'RIGHT']
        
        self.env_title = "LineWalking"
        
    def get_available_actions(self):
        return self.actions

    def agents_on_map(self):
        grid = np.zeros((self.width))
        grid[0,self.current_location] = 1
        return grid
    
    def get_reward(self, new_location):
        return self.cliff[new_location]
    
    def make_step(self, action):
        """moves the agent in the specified direction, unless the agent is 
        at a border, in which case the agent stays still"""
        last_location = self.current_location
        reward = 0
        if action == "LEFT":
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (0, self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)
        elif action == "RIGHT":
            if last_location[1] == self.width - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (0, self.current_location[1] + 1)
                reward = self.get_reward(self.current_location)
        return reward
    
    def check_state(self):
        """checks if the agent has reached a terminal state"""
        if self.current_location in self.terminal_states:
            print("STOPPED")
            return "TERMINAL"

# defining Gridworld environment
class GridWorld:
    def __init__(self, height = 5, width = 5):
        # defining grid world size
        self.height = height
        self.width = width
        self.grid = np.zeros((self.height, self.width)) - 1
        
        # set starting place for agent
        self.current_location = (4, np.random.randint(0,5))
        
        # set locations for bomb (negative reward) and gold (positive reward)
        self.bomb_location = (1,3)
        self.gold_location = (0,3)
        self.terminal_states = [self.bomb_location, self.gold_location]
        
        # defining reward values for terminal states
        self.grid[self.bomb_location[0], self.bomb_location[1]] = -10
        self.grid[self.gold_location[0], self.gold_location[1]] = 10
        
        # setting available actions for agent
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # environment name for graphing
        self.env_title = "GridWorld"
        
    def get_available_actions(self):
        # returns possible actions
        return self.actions
    
    def agents_on_map(self):
        """prints out a map representing the grid and the agent's location"""
        grid = np.zeros(( self.height, self.width))
        grid[self.current_location[0], self.current_location[1]] = 1
        return grid
    
    def get_reward(self, new_location):
        """returns the reward for an input position"""
        return self.grid[new_location[0], new_location[1]]
    
    def make_step(self, action):
        """moves the agent in the specified direction, unless the agent is 
        at a border, in which case the agent stays still"""
        last_location = self.current_location
        
        if action == "UP":
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
        elif action == "DOWN":
            if last_location[0] == self.height - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0] + 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
        elif action == "LEFT":
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0], self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)
        elif action == "RIGHT":
            if last_location[1] == self.width - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0], self.current_location[1] + 1)
                reward = self.get_reward(self.current_location)
        return reward
    
    def check_state(self):
        """checks if the agent has reached a terminal state"""
        if self.current_location in self.terminal_states:
            return "TERMINAL"
     
# defining CliffWalking environment
class CliffWalking:
    def __init__(self, height = 4, width = 12):
        # defining grid world size
        self.height = height
        self.width = width
        self.grid = np.zeros((self.height, self.width)) - 1
        # set starting place for agent
        self.start_location = (0,0)
        self.current_location = (0,0)
        
        # set locations for bomb (negative reward) and gold (positive reward)
        self.target_location = (0,11)
        self.terminal_states = [self.target_location]
        self.cliff_edges = [(0,x) for x in range(1,11)]
        # defining reward values for terminal states
        self.grid[self.target_location[0], self.target_location[1]] = 0
        for x in self.cliff_edges:
            self.grid[x[0], x[1]] = -100
        
        # setting available actions for agent
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # environment name for graphing
        self.env_title = "CliffWalking"
        
    def get_available_actions(self):
        # returns possible actions
        return self.actions
    
    def agents_on_map(self):
        """prints out a map representing the grid and the agent's location"""
        grid = np.zeros(( self.height, self.width))
        grid[self.current_location[0], self.current_location[1]] = 1
        return grid
    
    def get_reward(self, new_location):
        """returns the reward for an input position"""
        return self.grid[new_location[0], new_location[1]]
    
    def make_step(self, action):
        """moves the agent in the specified direction, unless the agent is 
        at a border, in which case the agent stays still
        agents that touch the cliff edge are teleported to the start"""
        last_location = self.current_location
        
        if action == "UP":
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
        elif action == "DOWN":
            if last_location[0] == self.height - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0] + 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
        elif action == "LEFT":
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0], self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)
        elif action == "RIGHT":
            if last_location[1] == self.width - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0], self.current_location[1] + 1)
                reward = self.get_reward(self.current_location)
        if self.current_location in self.cliff_edges:
            self.current_location = (0,0)
        return reward
    
    def check_state(self):
        """checks if the agent has reached a terminal state"""
        if self.current_location in self.terminal_states:
            return "TERMINAL"        
        
# defining an agent who simply chooses a random direction to move in
class RandomAgent():
    def choose_action(self, available_actions):
        return np.random.choice(available_actions)
    
    
# defining an agent who utilises Q-learning
# model-free RL algorithm that learns a policy for an agent
# agents has a table of all possible game states and actions for these states
# each pair of a state (S) and an action (A) has a reward Q
# for a given state, agent simply chooses the action that has the highest value
class QLearningAgent():
    # initialises 
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=1):
        self.environment = environment
        # table that stores the state and action pairs
        self.q_table = dict()
        for x in range(environment.height):
            for y in range(environment.width):
                # sets initial action rewards for a state
                self.q_table[(x,y)] = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}
        
        # % chance that the agent will perform a random exploratory action
        self.epsilon = epsilon
        # learning rate -> how much the difference between new values is changed
        self.alpha = alpha
        # discount factor -> used to balance future/immediate reward
        self.gamma = gamma
        
    def choose_action(self, available_actions):
        """Returns the optimal action for the state from the Q-Value table"""
        if np.random.uniform(0,1) < self.epsilon:
            # chooses a random exploratory action if chance is under epsilon
            action = np.random.choice(available_actions)
        else:
            # gets the actions associated with that state
            q_values_of_state = self.q_table[self.environment.current_location]
            # finds the maximum possible reward
            max_value = max(q_values_of_state.values())
            # chooses an action with the maximum possible reward
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == max_value])
        return action
    
    def learn(self, old_state, reward, new_state, action):
        """Updates the agent's q-value table using the Q-learning formula
        new value = old value + learning rate * (reward + discount factor * estimated optimal future value - old value)
        """
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]
        
        self.q_table[old_state][action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)
   

# implementation of a State-Action-Reward-State-Action (SARSA) agent
# on-policy control method
# learns an ACTION-VALUE function, rather than a STATE value function
# considers transitions from state-action pairs to state action pairs     
# learns based according to the agent's current policy, rather than the
# greedy policy approach of Q-learning
class SARSAAgent():
    def __init__(self, environment, epsilon=0.05,alpha=0.1, gamma=1):
        self.environment = environment
        self.q_table = dict()
        for x in range(environment.height):
            for y in range(environment.width):
                # sets initial action rewards for a state
                self.q_table[(x,y)] = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
    
    def choose_action(self, available_actions):
        if np.random.uniform(0,1) > self.epsilon:
            action = np.random.choice(available_actions)
        else:
            # gets the actions associated with that state
            q_values_of_state = self.q_table[self.environment.current_location]
            # finds the maximum possible reward
            max_value = max(q_values_of_state.values())
            # chooses an action with the maximum possible reward
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == max_value])
        return action
    
    def learn(self, old_state, reward, new_state,action):
        """chooses a new value using the policy, then calculates the new Q-value via
        old value + current_q_value * (reward + learning rate * NEW STATE Q VALUE - OLD STATE Q VALUE)"""
        next_action = self.choose_action(self.environment.get_available_actions())
        current_q_value = self.q_table[old_state][action]
        next_q_value = self.q_table[new_state][next_action]
        self.q_table[old_state][action] = current_q_value + self.alpha * (reward + self.gamma * next_q_value)
    
    
def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    """runs a number of episodes of the given environment"""
    reward_per_episode = []
    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over == False:
            old_state = environment.current_location
            action = agent.choose_action(environment.actions)
            reward = environment.make_step(action)
            new_state = environment.current_location
            
            if learn == True:
                agent.learn(old_state, reward, new_state, action)
                
            cumulative_reward += reward
            step += 1
            
            if environment.check_state() == "TERMINAL":
                print(f"Finished trial {trial} after {step} steps, reward {cumulative_reward}")
                environment.__init__()
                game_over = True
            
        reward_per_episode.append(cumulative_reward)
    return reward_per_episode

#environment = GridWorld()
#environment = LineWalking()
environment = CliffWalking()
random_agent = RandomAgent()
q_learning_agent = QLearningAgent(environment)
sarsa_agent = SARSAAgent(environment)

plt.title(f"{environment.env_title} reward values")
reward_per_episode = play(environment, random_agent, trials = 1000)
plt.plot(reward_per_episode, label="Random")
q_learning_reward_per_episode = play(environment, q_learning_agent, trials = 1000, learn = True)
plt.plot(q_learning_reward_per_episode, label="Q-Learning")
sarsa_learning_reward_per_episode = play(environment, sarsa_agent, trials = 1000, learn = True)
plt.plot(sarsa_learning_reward_per_episode, label="SARSA")
plt.legend(loc="lower left")
plt.show()