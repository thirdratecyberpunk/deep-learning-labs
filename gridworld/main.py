# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:09:17 2020

@author: Lewis
Implementing some reinforcement learning algorithms in Gridworld as defined by
https://github.com/michaeltinsley/Gridworld-with-Q-Learning-Reinforcement-Learning-/blob/master/Gridworld.ipynb
and LineWalking
"""

from environments.CliffWalking import CliffWalking
from agents.RandomAgent import RandomAgent
from agents.QLearningAgent import QLearningAgent
from agents.DeepQLearningAgent import DeepQLearningAgent
from agents.SARSAAgent import SARSAAgent

import matplotlib.pyplot as plt

import sys

def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False, update=False, target_update = 10):
    """runs a number of episodes of the given environment"""
    reward_per_episode = []
    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over == False:
            old_state = environment.current_location
            action = agent.choose_action(environment.actions)
#            print(f"Step {step}/ {max_steps_per_episode}: Agent chose {action}")
            reward = environment.make_step(action)
            new_state = environment.current_location
            if learn == True:
                agent.learn(old_state, reward, new_state, action)

            cumulative_reward += reward
            step += 1

            if environment.check_state() == "TERMINAL":
                environment.__init__()
                game_over = True

        print(f"Finished trial {trial}/{trials} after {step} steps, reward {cumulative_reward}")
        reward_per_episode.append(cumulative_reward)
        
        # tells agent to update, if that is relevant (i.e. dqn)
        if (update == True and trial % target_update == 0):
            agent.update()
    return reward_per_episode

def main():
    environment = CliffWalking()
    random_agent = RandomAgent()
    q_learning_agent = QLearningAgent(environment)
    deep_q_learning_agent = DeepQLearningAgent(environment)
    sarsa_agent = SARSAAgent(environment)
    
    plt.title(f"{environment.env_title} reward values")
#    random_reward_per_episode = play(environment, random_agent, trials = 1000)
#    plt.plot(random_reward_per_episode, label="Random")
#    q_learning_reward_per_episode = play(environment, q_learning_agent, trials = 1000, learn = True)
#    plt.plot(q_learning_reward_per_episode, label="Q-Learning")
    deep_q_learning_reward_per_episode = play(environment, deep_q_learning_agent, trials = 1000, learn = True, update=True)
    plt.plot(deep_q_learning_reward_per_episode, label="Deep Q-Learning")
#    sarsa_learning_reward_per_episode = play(environment, sarsa_agent, trials = 1000, learn = True)
#    plt.plot(sarsa_learning_reward_per_episode, label="SARSA")
    plt.legend(loc="lower left")
    plt.show()
    
if __name__ == "__main__":
    main()
