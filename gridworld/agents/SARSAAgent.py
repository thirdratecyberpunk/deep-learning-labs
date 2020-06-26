import numpy as np
import matplotlib.pyplot as plt

# implementation of a State-Action-Reward-State-Action (SARSA) agent
# on-policy control method
# learns an ACTION-VALUE function, rather than a STATE value function
# considers transitions from state-action pairs to state action pairs
# learns based according to the agent's current policy, rather than the
# greedy policy approach of Q-learning
# TODO: work out why this doesn't seem to learn correctly
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
