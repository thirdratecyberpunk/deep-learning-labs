import numpy as np
import matplotlib.pyplot as plt
from environments.Direction import Direction

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
                # TODO: make this defined from actions available in environment
                self.q_table[(x,y)] = {Direction.UP: 0, Direction.DOWN: 0, Direction.LEFT: 0, Direction.RIGHT: 0}

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
