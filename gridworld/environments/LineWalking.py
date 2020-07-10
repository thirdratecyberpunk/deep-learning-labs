# defining LineWalking
from environments.Direction import Direction

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

        self.actions = [Direction.LEFT, Direction.RIGHT]

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
        if action == Direction.LEFT:
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (0, self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)
        elif action == Direction.RIGHT:
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
