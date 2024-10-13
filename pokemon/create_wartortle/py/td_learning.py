# Import packages
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
import pokemon_env
import visualize_value_func

import numpy as np
import matplotlib.pyplot as plt
from pyboy import PyBoy

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from memory_addresses import *
from skimage.transform import resize
from IPython.display import clear_output
import cv2
import copy
import json
import yaml
import csv


def td0_learning(env, value_func, env_config, td_learning_config):
    """
    Run TD(0) to estimate state/action values
    """
    step_size = td_learning_config['td_learning']['step_size']
    epsilon = td_learning_config['td_learning']['epsilon']
    gamma = td_learning_config['td_learning']['gamma']
    num_episode = td_learning_config['td_learning']['num_episode']
    init_value = td_learning_config['td_learning']['init_value']
    max_steps = td_learning_config['td_learning']['max_steps']
    actions_list = []
    states_list = []
    rewards_list = []
    
    for i in range(num_episode):
        print(f"Episode {i+1}/{num_episode}")
        
        # Initialization
        state, _ = env.reset()
        terminated = False
        step_counter = 0
        states = []
        states.append(state)
        actions = []
        rewards = []

        # Add initial state
        value_func = initialize_new_state_action(env, state, value_func, init_value)
        
        while (not terminated)&(step_counter<=max_steps):
            # Choose an action with epsilon-greedy policy
            action = choose_epsilon_greedy_action(env, state, value_func, epsilon)
        
            # Take a step
            next_state, reward, terminated, truncated, info = env.step(action)

            # Add new states
            value_func = initialize_new_state_action(env, next_state, value_func, init_value)
                 
            # Append data
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            
            # Choose the next action with a greedy policy (A')
            next_action = choose_greedy_action(env, next_state, value_func)
            
            # Update the action value function
            value_func[state][action] = value_func[state][action] + step_size * (reward + gamma * value_func[next_state][next_action] - value_func[state][action])            

            # Update the current state and action
            state = next_state
            action = next_action
            
            if env_config['env']['rendering']:
                plt.imshow(env.render())
            
            step_counter += 1
            
        states_list.append(states)
        actions_list.append(actions)
        rewards_list.append(rewards)
            
    env.close()
    
    return value_func, states_list, actions_list, rewards_list


def tdn_learning(env, value_func, env_config, td_learning_config):
    """
    Run TD(N) learning to estimate state/action values
    """
    print(f"Episode {i+1}/{num_episode}")
    step_size = td_learning_config['td_learning']['step_size']
    epsilon = td_learning_config['td_learning']['epsilon']
    gamma = td_learning_config['td_learning']['gamma']
    num_episode = td_learning_config['td_learning']['num_episode']
    init_value = td_learning_config['td_learning']['init_value']
    max_steps = td_learning_config['td_learning']['max_steps']
    td_step = td_learning_config['td_learning']['td_step']  # New: TD(N) parameter
    
    actions_list = []
    states_list = []
    rewards_list = []
    
    for i in range(num_episode):
        # Initialization
        state, _ = env.reset()
        terminated = False
        step_counter = 0
        
        states = []
        actions = []
        rewards = []
        
        # Add initial state
        value_func = initialize_new_state_action(env, state, value_func, init_value)
        
        while (not terminated) & (step_counter <= max_steps):
            # Choose an action with epsilon-greedy policy
            action = choose_epsilon_greedy_action(env, state, value_func, epsilon)

            # Take a step
            next_state, reward, terminated, truncated, info = env.step(action)

            # Add new states
            value_func = initialize_new_state_action(env, next_state, value_func, init_value)

            # Append data
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # If we have collected N steps or the episode terminates, update the value function
            if len(states) >= td_step or terminated:
                # Compute the N-step return
                G = 0
                for j in range(len(rewards)):
                    G += (gamma ** j) * rewards[j]  # Discounted rewards up to N steps

                if not terminated and len(states) == td_step:
                    # Add the estimate of the future value (bootstrapping) if not terminal
                    next_action = choose_greedy_action(env, next_state, value_func)
                    G += (gamma ** td_step) * value_func[next_state][next_action]

                # Update the value function using the N-step return
                value_func[states[0]][actions[0]] = value_func[states[0]][actions[0]] + step_size * (G - value_func[states[0]][actions[0]])

                # Shift the states and actions window (for overlapping updates)
                states.pop(0)
                actions.pop(0)
                rewards.pop(0)
            
            # Update the current state
            state = next_state
            step_counter += 1
            
            if env_config['env']['rendering']:
                plt.imshow(env.render())
        
        # Store episode details
        states_list.append(states)
        actions_list.append(actions)
        rewards_list.append(rewards)
    
    env.close()
    return value_func, states_list, actions_list, rewards_list


def choose_epsilon_greedy_action(env, state, value_func, epsilon):
    """
    Choose an action based on an epsilon-greedy policy.
    """
    p = np.random.random()  # Generate a random number in the range (0, 1).
    if p < epsilon:
        action = np.random.choice(env.action_space.n) # Random action
    else:
        action_values_at_s = value_func[state]
        max_value = max(action_values_at_s.values())
        greedy_actions = [k for k, v in action_values_at_s.items() if v == max_value]
        action = np.random.choice(greedy_actions) # Greedy action
    return action


def choose_greedy_action(env, state, value_func):
    """
    Choose an action based on an greedy policy. (tie break randomly)
    """
    action_values_at_s = value_func[state]
    max_value = max(action_values_at_s.values())
    greedy_actions = [k for k, v in action_values_at_s.items() if v == max_value]
    action = np.random.choice(greedy_actions) # Greedy action

    return action


def choose_softmax_action(env, state, value_func, temperature=1.0):
    """
    Choose an action based on softmax action selection.
    """
    action_values_at_s = value_func[state]
    action_values = np.array(list(action_values_at_s.values()))
    
    # Apply temperature scaling: controls exploration (higher temperature -> more exploration)
    scaled_action_values = action_values / temperature
    
    # Compute softmax probabilities
    exp_values = np.exp(scaled_action_values - np.max(scaled_action_values))  # Stability with max subtraction
    softmax_probs = exp_values / np.sum(exp_values)
    
    # Choose an action based on the computed probabilities
    actions = list(action_values_at_s.keys())
    action = np.random.choice(actions, p=softmax_probs)
    
    return action


def initialize_new_state_action(env, state, value_func, init_value):
    """
    Initialize the value of new state and action pairs
    """
    if state not in value_func.keys():
        value_func[state] = {}
        for action_ind in range(len(env.valid_actions)):
            value_func[state][action_ind] = init_value
        
    return value_func


def test(env, value_func, td_learning_config):
    # Initialization
    state, _ = env.reset()
    terminated = False
    states = []
    states.append(copy.deepcopy(env.render()))
    actions = []
    rewards = []
    init_value = td_learning_config['td_learning']['init_value']
    test_step = td_learning_config['output'].get('test_step', 100)
    action_selection_algo = td_learning_config['output'].get('test_action_selec_algo', 'greedy')
    step_count = 0

    while (not terminated)&(step_count<=test_step):
        if action_selection_algo=='greedy':
            action = int(choose_greedy_action(env, state, value_func))
        elif action_selection_algo=='epsilon_greedy':
            action = int(choose_epsilon_greedy_action(env, state, value_func, epsilon=0.1))
        elif action_selection_algo=='softmax':
            action = int(choose_softmax_action(env, state, value_func, temperature=0.5))
    
        # Take a step
        next_state, reward, terminated, truncated, info = env.step(action)
        states.append(copy.deepcopy(env.render()))
        rewards.append(reward)
        
        # Add new states
        value_func = initialize_new_state_action(env, next_state, value_func, init_value)

        # Update the current state and action
        state = next_state
        
        # plt.imshow(env.render())
        step_count += 1
    
    env.close()
    
    if td_learning_config['output']['save_video']:
        output_mp4(states, td_learning_config['output']['video_path'])
    
    return rewards
      
      
def output_mp4(states, file_path, image_size=(160,144)):
    images = states.copy()
    out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, image_size)
    for frame in images:
        out.write(frame[:,:,:3]) # frame is a numpy.ndarray with shape (160, 144, 3)
    out.release()
  
    
def main(env_config, td_learning_config, vis_config, td_type='td0'):
    env = pokemon_env.RedGymEnv(env_config)

    # Initialize the action value function
    print('Run training')
    value_func = {}
    if td_type=='td0':
        # Run TD(0) to estimate the action values
        value_func, states_list, actions_list, rewards_list = td0_learning(env, value_func, env_config, td_learning_config)
    elif td_type=='tdn':
        # Run TD(n) to estimate the action values
        value_func, states_list, actions_list, rewards_list = tdn_learning(env, value_func, env_config, td_learning_config)

    # Save a learned value function
    with open(td_learning_config['output']['value_func_path'], 'w') as f:
        json.dump(value_func, f)

    # Save logs
    if td_learning_config['output']['save_log']:
        with open(td_learning_config['output']['state_log_path'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(states_list)
        with open(td_learning_config['output']['action_log_path'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(actions_list)
        with open(td_learning_config['output']['reward_log_path'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(rewards_list)

    # Visualize learned value function
    if td_learning_config['output']['save_value_func_vis']:
        visualize_value_func.main(vis_config, td_learning_config['output']['value_func_path'], td_learning_config['output']['value_func_vis_path'])

    if td_learning_config['output']['run_test']:
        print('Run test')
        # Test
        env = pokemon_env.RedGymEnv(env_config)
        with open(td_learning_config['output']['value_func_path']) as f:
            value_func = json.load(f)
        test(env, value_func, td_learning_config)
