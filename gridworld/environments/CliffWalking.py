# defining CliffWalking environment
import numpy as np
import matplotlib.pyplot as plt

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
