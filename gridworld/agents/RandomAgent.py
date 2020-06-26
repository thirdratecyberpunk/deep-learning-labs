import numpy as np
import matplotlib.pyplot as plt

# defining an agent who simply chooses a random direction to move in
class RandomAgent():
    def choose_action(self, available_actions):
        return np.random.choice(available_actions)
