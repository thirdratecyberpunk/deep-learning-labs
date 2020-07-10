# defining Gridworld environment
from environments.Direction import Direction

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
        self.actions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

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

        if action == Direction.UP:
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
        elif action == Direction.DOWN:
            if last_location[0] == self.height - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0] + 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
        elif action == Direction.LEFT:
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (self.current_location[0], self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)
        elif action == Direction.RIGHT:
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
