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
import argparse

import torch
import numpy as np
import random

import datetime

def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False, update=False, target_update = 10):
    """runs a number of episodes of the given environment"""
    reward_per_episode = []
    steps_per_episode = []
    for trial in range(trials):
        cumulative_reward = 0
        total_steps = 0
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
            total_steps += 1

            if environment.check_state() == "TERMINAL":
                environment.__init__()
                game_over = True
                
            # tells agent to update, if that is relevant (i.e. dqn)
            if (update == True and total_steps % target_update == 0):
                agent.update()

        print(f"Finished trial {trial}/{trials} after {step} steps, reward {cumulative_reward}")
        reward_per_episode.append(cumulative_reward)
        steps_per_episode.append(total_steps)
    return reward_per_episode, steps_per_episode

# TODO: x + y + z assumes three lists, need to work out how to fix this
def generate_average_list_from_list_of_lists(list_to_average):
    return [(x + y + z) / len(list_to_average) for x, y, z in zip(*list_to_average)]

def main():
    # sorting command line arguments
    parser = argparse.ArgumentParser(description='Demonstration of RL algorithms in Gridworld style environments')
    parser.add_argument(
            '--seeds',
            nargs="*",
            type=int,
            default=[0,1,2],
            help='List containing at least one seed')
    parser.add_argument(
            '--trials',
            type=int,
            default=2000,
            help='Number of trials to run')
    parser.add_argument(
            '--max_steps_per_episode',
            type=int,
            default=1000,
            help='Number of steps per trial'
            )
    parser.add_argument(
            '--output',
            default='output/',
            help='Location to save .csv file of results to')
    
    args = parser.parse_args()
    
    # setting date/time for chart output
    d = datetime.datetime.now()
    
    random_all_rewards_per_episode = []
    q_all_rewards_per_episode = []
    sarsa_all_rewards_per_episode = []
    dq_all_rewards_per_episode = []
    
    random_all_steps_per_episode = [] 
    q_all_steps_per_episode = [] 
    sarsa_all_steps_per_episode = []
    dq_all_steps_per_episode = [] 
    
    # running experiments and plotting results on a graph
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # setting up environment and agents
        environment = CliffWalking()
        random_agent = RandomAgent()
        q_learning_agent = QLearningAgent(environment)
        deep_q_learning_agent = DeepQLearningAgent(environment)
        sarsa_agent = SARSAAgent(environment)
        
        # running experiments
        random_reward_per_episode, random_total_steps = play(environment, random_agent, trials = args.trials, max_steps_per_episode= args.max_steps_per_episode)
#        q_learning_reward_per_episode, q_learning_total_steps = play(environment, q_learning_agent, trials = args.trials,
#                                                                     max_steps_per_episode= args.max_steps_per_episode, learn = True)
        sarsa_learning_reward_per_episode, sarsa_total_steps = play(environment, sarsa_agent, trials =args.trials, learn = True)
#        deep_q_learning_reward_per_episode, deep_q_learning_total_steps = play(environment, deep_q_learning_agent, trials = args.trials,
#                                                                               max_steps_per_episode= args.max_steps_per_episode, 
#                                                                               learn = True, update=True)
        
        random_all_rewards_per_episode.append(random_reward_per_episode)
        random_all_steps_per_episode.append(random_total_steps)
        
#        q_all_rewards_per_episode.append(q_learning_reward_per_episode)
#        q_all_steps_per_episode.append(q_learning_total_steps)
        
        sarsa_all_rewards_per_episode.append(sarsa_learning_reward_per_episode)
        sarsa_all_steps_per_episode.append(sarsa_total_steps)
        
#        dq_all_rewards_per_episode.append(deep_q_learning_reward_per_episode)
#        dq_all_steps_per_episode.append(deep_q_learning_total_steps)
        
        # diagram displaying loss over time
        fig = plt.figure()
        plt.title(f"{environment.env_title} reward values for seed {seed}")
        plt.xlabel("Trial")
        plt.ylabel("Loss")
        plt.plot(random_reward_per_episode, label="Random")
#        plt.plot(q_learning_reward_per_episode, label="Q-Learning")
        plt.plot(sarsa_learning_reward_per_episode, label="SARSA")
#        plt.plot(deep_q_learning_reward_per_episode, label="Deep Q-Learning")
        plt.legend(loc="lower left")
        plt.show()
        fig.savefig(f"{args.output}/environment_{environment.env_title}_seed_{seed}_loss_results_{d:%Y-%m-%d_%H-%M-%S}.png")
        
        # diagram displaying number of steps taken over time
        fig = plt.figure()
        plt.title(f"{environment.env_title} number of steps taken per trial for seed {seed}")
        plt.xlabel("Trial")
        plt.ylabel("Number of steps taken")
        plt.plot(random_total_steps, label="Random")
#        plt.plot(q_learning_total_steps, label="Q-Learning")
        plt.plot(sarsa_total_steps, label="SARSA")
#        plt.plot(deep_q_learning_total_steps, label="Deep Q-Learning")
        plt.legend(loc="lower left")
        plt.show()
        fig.savefig(f"{args.output}/environment_{environment.env_title}_seed_{seed}_steps_taken_results_{d:%Y-%m-%d_%H-%M-%S}.png")
    
    # generating average performance over seeds
    average_random_rewards = generate_average_list_from_list_of_lists(random_all_rewards_per_episode)
    average_random_steps = generate_average_list_from_list_of_lists(random_all_steps_per_episode)
#    average_q_learning_rewards = generate_average_list_from_list_of_lists(q_all_rewards_per_episode)
#    average_q_learning_steps = generate_average_list_from_list_of_lists(q_all_steps_per_episode)
    average_sarsa_learning_rewards = generate_average_list_from_list_of_lists(sarsa_all_rewards_per_episode)
    average_sarsa_steps = generate_average_list_from_list_of_lists(sarsa_all_steps_per_episode)
#    average_dq_learning_rewards = generate_average_list_from_list_of_lists(dq_all_rewards_per_episode)
#    average_dq_learning_steps = generate_average_list_from_list_of_lists(dq_all_steps_per_episode)
    
    # diagram displaying seed average reward over time
    fig = plt.figure()
    plt.title(f"{environment.env_title} reward values for average of seeds {args.seeds}")
    plt.xlabel("Trial")
    plt.ylabel("Loss")
    plt.plot(average_random_rewards, label="Random")
#    plt.plot(average_q_learning_rewards, label="Q-Learning")
    plt.plot(average_sarsa_learning_rewards, label="SARSA")
#    plt.plot(average_dq_learning_rewards, label="Deep Q-Learning")
    plt.legend(loc="lower left")
    plt.show()
    fig.savefig(f"{args.output}/environment_{environment.env_title}_average_loss_results_{d:%Y-%m-%d_%H-%M-%S}.png")
    
    # diagram displaying seed average steps taken over time
    fig = plt.figure()
    plt.title(f"{args.output}/{environment.env_title} steps taken per trial for average of seeds {args.seeds}")
    plt.xlabel("Trial")
    plt.ylabel("Number of steps taken")
    plt.plot(average_random_steps, label="Random")
#    plt.plot(average_q_learning_steps, label="Q-Learning")
    plt.plot(average_sarsa_steps, label="SARSA")
#    plt.plot(average_dq_learning_steps, label="Deep Q-Learning")
    plt.legend(loc="lower left")
    plt.show()
    fig.savefig(f"{args.output}/environment_{environment.env_title}_average_steps_results_{d:%Y-%m-%d_%H-%M-%S}.png")
    
if __name__ == "__main__":
    main()
