import numpy as np
import collections
from collections import deque
import gymnasium as gym
import itertools
import random
import os
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Environment MuZero is interacting with
env = gym.make('CartPole-v0')

class Node(object):
    
    def __init__(self, prior):
        """
        Node in MCTS
        prior: The prior policy on the node, computed from policy network
        """
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_representation = None
        self.reward = 0
        self.expanded = False

    def value(self):
        """
        Compute expected value of a node
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count
        
        
class Game:
    """
    A single episode of interaction with the environment.
    """
    def __init__(self, action_space_size, discount, curr_state):

        self.action_space_size = action_space_size
        self.curr_state = curr_state
        self.done = False
        self.discount = discount
        self.priorities = None

        self.state_history = [self.curr_state]
        self.action_history = []
        self.reward_history = []

        self.root_values = []
        self.child_visits = []

    def store_search_statistics(self, root):
        """
        Stores the search statistics for the current root node
        
        root: Node object including the infomration of the current root node
        """        
        # Stores the normalized root node child visits (i.e. policy target)
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append(np.array([
            root.children[a].visit_count
            / sum_visits if a in root.children else 0
            for a in range(self.action_space_size)
        ]))
        
        # Stores the root node value, computed from the MCTS (i.e. vlaue target)
        self.root_values.append(root.value())

    def take_action(self, action, env):
        """
        Take an action and store the action, reward, and new state into history
        """
        observation, reward, terminated, truncated, _ = env.step(action)
        self.curr_state = observation
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.done = terminated | truncated
        if not self.done:
            self.state_history.append(self.curr_state)

    def make_target(self, state_index, num_unroll_steps, td_steps):
        """
        Makes the target data for training

        state_index: the start state
        num_unroll_steps: how many times to unroll from the current state
                          each unroll forms a new target
        td_steps: the number of td steps used in bootstrapping the value function
        """
        targets = [] # target = (value, reward, policy)
        actions = []

        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps

            # target value of the current node is the sum of 1) discounted rewards up to bootstrap index + 2) discounted value at bootstrap index            
            
            # compute 2)
            # assign value=0 if bootstrap_index is after the end of episode
            # otherwise, assign discounted value at bootstrap_index state
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index][0] * (self.discount**td_steps)
            else:
                value = 0
            
            # compute 1)  
            # add discounted reward values earned between current_index and bootstrap_index
            for i, reward in enumerate(self.reward_history[current_index:bootstrap_index]):
                value += reward * (self.discount**i)

            # if current_index is after the end of episode, assign 0 as last_reward
            # otherwise, assign the reward from last step as last_reward, which will be used as reward target
            if current_index > 0 and current_index <= len(self.reward_history):
                last_reward = self.reward_history[current_index-1]
            else:
                last_reward = 0
                
            if current_index < len(self.root_values): # current_index is within the episode, 
                targets.append((value, last_reward,
                                self.child_visits[current_index]))
                actions.append(self.action_history[current_index])
            else: # current_index is after the end of episode
                # State which pasts the end of the game are treated as an absorbing state.
                num_actions = self.action_space_size
                # targets.append((0, last_reward, []))
                targets.append(
                    (0, last_reward, np.array([1.0 / num_actions for _ in range(num_actions)]))) # assign value 0 and uniform policy
                actions.append(np.random.choice(num_actions)) # assign a random action
        return targets, actions


class ReplayBuffer(object):
    """
    Store training data acquired through self-play
    """
    def __init__(self, config):
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.buffer = deque(maxlen=self.buffer_size) # deque: list-like container with fast appends and pops on either end
        self.td_steps = config['td_steps']
        self.unroll_steps = config['num_unroll_steps']

    def save_game(self, game):
        """
        Save a game into replay buffer.
        Max number of games saved in the buffer is defined as self.buffer_size
        """
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        """
        Sample batch_size games, along with an associated start position in each game
        Make the targets for the batch to be used in training
        """
        games = [self.sample_game() for _ in range(self.batch_size)] # randomly sample batch_size games
        game_pos = [self.sample_position(g) for g in games] # randomly sample position from the game
        batch = []
        for (g, i) in zip(games, game_pos):
            # create training targets (output) and actions (input)
            targets, actions = g.make_target(
                i, self.unroll_steps, self.td_steps) # each target = (value, reward, policy)
            batch.append(
                (g.state_history[i], actions, targets))
        state_batch, actions_batch, targets_batch = zip(*batch) # unpack batch
        actions_batch = list(zip(*actions_batch)) # unpack action
        targets_init_batch, *targets_recurrent_batch = zip(*targets_batch) # unpack targets_batch, targets_init_batch: initial target, targets_recurrent_batch: subsequent targets
        # * operator is used for extended unpacking, meaning that any additional targets beyond the initial one are packed into targets_recurrent_batch.
        batch = (state_batch, targets_init_batch, targets_recurrent_batch,
                 actions_batch)

        return batch

    def sample_game(self):
        """
        Ramdonly sample a game from buffer
        """
        game = np.random.choice(self.buffer)
        return game

    def sample_position(self, game):
        """
        Randomply sample position from a game to start unrolling
        """
        sampled_index = np.random.randint(
            len(game.reward_history)-self.unroll_steps) # limit the sample in the space where we can unroll # of unroll_steps
        return sampled_index
    

class RepresentationNetwork(nn.Module):
    """
    Input: raw state of the current root
    Output: latent state of the current root
    """
    def __init__(self, input_size, hidden_neurons, embedding_size):
        super(RepresentationNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, embedding_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
    
    
class ValueNetwork(nn.Module):
    """
    Input: latent state
    Output: expected value at the input latent state
    """
    def __init__(self, input_size, hidden_neurons, value_support_size):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, value_support_size)
        )

    def forward(self, x):
        return self.layers(x)
    
    
class PolicyNetwork(nn.Module):
    """
    Input: latent state
    Output: policy at the input latent state
    """
    def __init__(self, input_size, hidden_neurons, action_size):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, action_size)
        )

    def forward(self, x):
        return self.layers(x)
    
    
class DynamicNetwork(nn.Module):
    """
    Input: latent state & action to take
    Output: next latent state
    """
    def __init__(self, input_size, hidden_neurons, embedding_size):
        super(DynamicNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, embedding_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
    
    
class RewardNetwork(nn.Module):
    """
    Input: latent state & action to take
    Output: expected immediate reward
    """
    def __init__(self, input_size, hidden_neurons):
        super(RewardNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)
    
    
class InitialModel(nn.Module):
    """
    Combine Representation, Value, and Policy networks
    """
    def __init__(self, representation_network, value_network, policy_network):
        super(InitialModel, self).__init__()
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network

    def forward(self, state):
        hidden_representation = self.representation_network(state)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, value, policy_logits
    

class RecurrentModel(nn.Module):
    """
    Combine Dynamic, Reward, Value, and Policy network
    """
    def __init__(self, dynamic_network, reward_network, value_network, policy_network):
        super(RecurrentModel, self).__init__()
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.value_network = value_network
        self.policy_network = policy_network

    def forward(self, state_with_action):
        hidden_representation = self.dynamic_network(state_with_action)
        reward = self.reward_network(state_with_action)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, reward, value, policy_logits
    
    
class Networks(nn.Module):
    """
    Create both InitialModel and RecurrentModel class objects 
    and helper functions to run MCTS and train models
    """
    def __init__(self, representation_network, value_network, policy_network, dynamic_network, reward_network, max_value):
        super().__init__()
        self.train_steps = 0
        self.action_size = 2
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.initial_model = InitialModel(self.representation_network, self.value_network, self.policy_network)
        self.recurrent_model = RecurrentModel(self.dynamic_network, self.reward_network, self.value_network,
                                              self.policy_network)
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1

    def initial_inference(self, state):
        hidden_representation, value, policy_logits = self.initial_model(state)
        assert isinstance(self._value_transform(value), float)
        return self._value_transform(value), 0, policy_logits, hidden_representation

    def recurrent_inference(self, hidden_state, action):
        hidden_state_with_action = self._hidden_state_with_action(hidden_state, action)
        hidden_representation, reward, value, policy_logits = self.recurrent_model(hidden_state_with_action)
        return self._value_transform(value), self._reward_transform(reward), policy_logits, hidden_representation

    def _value_transform(self, value_support):
        """
        Apply invertable transformation to get a numpy scalar value
        """
        epsilon = 0.001
        value = torch.nn.functional.softmax(value_support)
        value = np.dot(value.detach().numpy(), range(self.value_support_size))
        value = np.sign(value) * (
                ((np.sqrt(1 + 4 * epsilon
                 * (np.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1
        )
        return value

    def _reward_transform(self, reward):
        """
        Transform reward into a numpy scalar value
        """
        return reward.detach().cpu().numpy()  # Assuming reward is a PyTorch tensor

    def _hidden_state_with_action(self, hidden_state, action):
        """
        Merge hidden state and one hot encoded action
        """
        hidden_state_with_action = torch.concat(
            (hidden_state, torch.tensor(self._action_to_one_hot(action, self.action_size))[0]), axis=0)
        return hidden_state_with_action
    
    def _action_to_one_hot(self, action, action_space_size):
        """
        Compute one hot of action to be combined with state representation
        """
        return np.array([1 if i == action else 0 for i in range(action_space_size)]).reshape(1, -1)
    
    def _scalar_to_support(self, target_value):
        """
        Transform value into a multi-dimensional target value to train a network
        """
        batch = target_value.size(0)
        targets = torch.zeros((batch, self.value_support_size))
        target_value = torch.sign(target_value) * \
            (torch.sqrt(torch.abs(target_value) + 1)
            - 1 + 0.001 * target_value)
        target_value = torch.clamp(target_value, 0, self.value_support_size)
        floor = torch.floor(target_value)
        rest = target_value - floor
        targets[torch.arange(batch, dtype=torch.long), floor.long()] = 1 - rest
        indexes = floor.long() + 1
        mask = indexes < self.value_support_size
        batch_mask = torch.arange(batch)[mask]
        rest_mask = rest[mask]
        index_mask = indexes[mask]
        targets[batch_mask, index_mask] = rest_mask
        return targets



def scale_gradient(tensor, scale):
    """
    Function to scale gradient as described in MuZero Appendix
    """
    return tensor * scale + tensor.detach() * (1. - scale)


def train_network(config, network, replay_buffer, optimizer, train_results):
    """
    Train Networks
    """
    for _ in range(config['train_per_epoch']):
        batch = replay_buffer.sample_batch()
        update_weights(config, network, optimizer, batch, train_results)


def update_weights(config, network, optimizer, batch, train_results):
    """
    Train networks by sampling games from repay buffer
    config: dictionary specifying parameter configurations
    network: network class to train
    optimizer: optimizer used to update the network_model weights
    batch: batch of experience
    train_results: class to store the train results
    """
    # for every game in sample batch, unroll and update network_model weights
    def loss():
        mse = torch.nn.MSELoss()

        loss = 0
        total_value_loss = 0
        total_reward_loss = 0
        total_policy_loss = 0
        (state_batch, targets_init_batch, targets_recurrent_batch,
         actions_batch) = batch

        state_batch = torch.tensor(state_batch)

        # get prediction from initial model (i.e. combination of dynamic, value, and policy networks)
        hidden_representation, initial_values, policy_logits = network.initial_model(state_batch)

        # create a value and policy target from batch data
        target_value_batch, _, target_policy_batch = zip(*targets_init_batch) # (value, reward, policy)
        target_value_batch = torch.tensor(target_value_batch).float()
        target_value_batch = network._scalar_to_support(target_value_batch) # transform into a multi-dimensional target

        # compute the error for the initial inference
        # reward error is always 0 for initial inference
        value_loss = F.cross_entropy(initial_values, target_value_batch)
        policy_loss = F.cross_entropy(policy_logits, torch.tensor(target_policy_batch))
        loss = 0.25 * value_loss + policy_loss

        total_value_loss = 0.25 * value_loss.item()
        total_policy_loss = policy_loss.item()

        # unroll batch with recurrent inference and accumulate loss
        for actions_batch, targets_batch in zip(actions_batch, targets_recurrent_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(*targets_batch)

            # get prediction from recurrent_model (i.e. dynamic, reward, value, and policy networks)
            actions_batch_onehot = F.one_hot(torch.tensor(actions_batch), num_classes=network.action_size).float()
            state_with_action = torch.cat((hidden_representation, actions_batch_onehot), dim=1)
            hidden_representation, rewards, values, policy_logits = network.recurrent_model(state_with_action)

            # create a value, policy, and reward target from batch data
            target_value_batch = torch.tensor(target_value_batch).float()
            target_value_batch = network._scalar_to_support(target_value_batch)
            target_policy_batch = torch.tensor(target_policy_batch).float()
            target_reward_batch = torch.tensor(target_reward_batch).float()

            # compute the loss for recurrent_inference 
            value_loss = F.cross_entropy(values, target_value_batch)
            policy_loss = F.cross_entropy(policy_logits, target_policy_batch)
            reward_loss = mse(rewards, target_reward_batch)

            # accumulate loss
            loss_step = (0.25 * value_loss + reward_loss + policy_loss)
            total_value_loss += 0.25 * value_loss.item()
            total_policy_loss += policy_loss.item()
            total_reward_loss += reward_loss.item()
                        
            # gradient scaling
            gradient_loss_step = scale_gradient(loss_step,(1/config['num_unroll_steps']))
            loss += gradient_loss_step
            scale = 0.5
            hidden_representation = hidden_representation / scale

        # store loss result for plotting
        train_results.total_losses.append(loss.item())
        train_results.value_losses.append(total_value_loss)
        train_results.policy_losses.append(total_policy_loss)
        train_results.reward_losses.append(total_reward_loss)
        return loss

    optimizer.zero_grad()
    loss = loss()
    loss.backward() # Compute gradients of loss with respect to parameters
    optimizer.step() # Update parameters based on gradients
    network.train_steps += 1    
    
    
class MCTS():
    
    def __init__(self, config):
        self.config = config
        
    def run_mcts(self, config, root, network, min_max_stats):
        """
        Run the main loop of MCTS for config['num_simulations'] simulations

        root: the root node
        network: the network
        min_max_stats: the min max stats object
        """
        for i in range(config['num_simulations']):
            history = []
            node = root
            search_path = [node]

            # expand node until reaching the leaf node
            while node.expanded:
                action, node = self.select_child(config, node, min_max_stats)
                history.append(action)
                search_path.append(node)
            parent = search_path[-2]
            action = history[-1]
            
            # expand the leaf node
            value = self.expand_node(node, list(
                range(config['action_space_size'])), network, parent.hidden_representation, action)
            
            # perform backpropagation
            self.backpropagate(search_path, value,
                        config['discount'], min_max_stats)


    def select_action(self, config, node, test=False):
        """
        Select an action to take
        train mode: action selection is performed stochastically (softmax)
        test mode: action selection is performed deterministically (argmax)
        """
        visit_counts = [
            (child.visit_count, action) for action, child in node.children.items()
        ]
        if not test:
            t = config['visit_softmax_temperature_fn']
            action = self.softmax_sample(visit_counts, t)
        else:
            action = self.softmax_sample(visit_counts, 0)
        return action


    def select_child(self, config, node, min_max_stats):
        """
        Select a child at an already expanded node
        Selection is based on the UCB score
        """
        best_action, best_child = None, None
        ucb_compare = -np.inf
        for action,child in node.children.items():
            ucb = self.ucb_score(config, node, child, min_max_stats)
            if ucb > ucb_compare:
                ucb_compare = ucb
                best_action = action # action, int
                best_child = child # node object
        return best_action, best_child


    def ucb_score(self, config, parent, child, min_max_stats):
        """
        Compute UCB Score of a child given the parent statistics
        Appendix B of MuZero paper
        """
        pb_c = np.log((parent.visit_count + config['pb_c_base'] + 1)
                    / config['pb_c_base']) + config['pb_c_init']
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c*child.prior.detach().numpy()
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(
                child.reward + config['discount']*child.value())
        else:
            value_score = 0
        return prior_score + value_score


    def expand_root(self, node, actions, network, current_state):
        """
        Expand the root node given the current state
        """
        # obtain the latent state, policy, and value of the root node 
        # by using a InitialModel
        observation = torch.tensor(current_state)
        transformed_value, reward, policy_logits, hidden_representation = network.initial_inference(observation)
        node.hidden_representation = hidden_representation
        node.reward = reward # always 0 for initial inference

        # extract softmax policy and set node.policy
        softmax_policy = torch.nn.functional.softmax(torch.squeeze(policy_logits))
        node.policy = softmax_policy

        # instantiate node's children with prior values, obtained from the predicted policy
        for action, prob in zip(actions, softmax_policy):
            child = Node(prob)
            node.children[action] = child

        # set node as expanded
        node.expanded = True
        
        return transformed_value


    def expand_node(self, node, actions, network, parent_state, parent_action):
        """
        Expand a leaf node given the parent state and action
        """
        # run recurrent inference at the leaf node
        transformed_value, reward, policy_logits, hidden_representation = network.recurrent_inference(parent_state, parent_action)
        node.hidden_representation = hidden_representation
        node.reward = reward

        # compute softmax policy and store it to node.policy
        softmax_policy = torch.nn.functional.softmax(torch.squeeze(policy_logits))
        node.policy = softmax_policy

        # instantiate node's children with prior values, obtained from the predicted softmax policy
        for action, prob in zip(actions,softmax_policy):
            child = Node(prob)
            node.children[action] = child

        # set node as expanded
        node.expanded = True
        
        return transformed_value


    def backpropagate(self, path, value, discount, min_max_stats):
        """
        Update a discounted total value and total visit count
        """
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value 
            min_max_stats.update(node.value())
            value = node.reward + discount * value


    def softmax_sample(self, visit_counts, temperature):
        """
        Sample an action
        """
        counts_arr = np.array([c[0] for c in visit_counts])
        if temperature == 0: # argmax
            action_idx = np.argmax(counts_arr)
        else: # softmax
            numerator = np.power(counts_arr,1/temperature)
            denominator = np.sum(numerator)
            dist = numerator / denominator
            action_idx = np.random.choice(np.arange(len(counts_arr)),p=dist)

        return action_idx
    
    
    def add_exploration_noise(self, config, node):
        """
        Add exploration noise by adding dirichlet noise to the prior over children
        This is governed by root_dirichlet_alpha and root_exploration_fraction
        """
        actions = list(node.children.keys())
        noise = np.random.dirichlet([config['root_dirichlet_alpha']]*len(actions))
        frac = config['root_exploration_fraction']
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1-frac) + n*frac

    
    
class MinMaxStats(object):
    """
    Store the min-max values of the environment to normalize the values
    Max value will be 1 and min value will be 0
    """

    def __init__(self, minimum, maximum):
        self.maximum = maximum
        self.minimum = minimum

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value



class TrainResults(object):
    def __init__(self):
        self.value_losses = []
        self.reward_losses = []
        self.policy_losses = []
        self.total_losses = []

    def plot_total_loss(self):
        x_vals = np.arange(len(self.total_losses))
        fig, ax = plt.subplots()
        ax.plot(x_vals, self.total_losses, label="Train Loss")
        plt.xlabel("Train Steps")
        plt.ylabel("Loss")
        plt.savefig('./RL/ModelBasedML/figure/total_loss.png')

    def plot_individual_losses(self):
        x_vals = np.arange(len(self.total_losses))
        fig, ax = plt.subplots()
        ax.plot(x_vals, self.value_losses, label="Value Loss")
        ax.plot(x_vals, self.policy_losses, label="Policy Loss")
        ax.plot(x_vals, self.reward_losses, label="Reward Loss")
        plt.xlabel("Train Steps")
        plt.ylabel("Losses")
        plt.legend()
        plt.savefig('./RL/ModelBasedML/figure/individual_loss.png')


class TestResults(object):

    def __init__(self):
        self.test_rewards = []

    def add_reward(self, reward):
        self.test_rewards.append(reward)

    def plot_rewards(self):
        x_vals = np.arange(len(self.test_rewards))
        fig, ax = plt.subplots()
        ax.plot(x_vals, self.test_rewards, label="Test Reward")
        plt.xlabel("Test Episodes")
        plt.ylabel("Reward")
        plt.savefig('./RL/ModelBasedML/figure/test_reward.png')


def self_play(env, config, replay_buffer, network):
    # create objects to store data for plotting
    test_rewards = TestResults()
    train_results = TrainResults()
    
    # create optimizer for training
    optimizer = torch.optim.Adam(network.parameters(), lr=config['lr_init'])
    
    # self-play and network training iterations
    for i in range(config['num_epochs']):  # Number of Steps of train/play alternations
        print(f"===Epoch Number {i}===")
        score = play_games(
            config, replay_buffer, network, env)
        print("Average traininig score:", score)
        train_network(config, network, replay_buffer, optimizer, train_results)
        print("Average test score:", test(config, network, env, test_rewards))

    # plot
    train_results.plot_individual_losses()
    train_results.plot_total_loss()
    test_rewards.plot_rewards()


def play_games(config, replay_buffer, network, env):
    """
    Play multiple games and store them in the replay buffer
    """
    returns = 0

    for _ in range(config['games_per_epoch']):
        game = play_game(config, network, env)
        replay_buffer.save_game(game)
        returns += sum(game.reward_history)

    return returns / config['games_per_epoch']


def play_game(config, network: Networks, env):
    """
    Plays one game
    """
    # Initialize environment
    start_state, _ = env.reset()
    
    game = Game(config['action_space_size'], config['discount'], start_state)        
    mcts = MCTS(config)
    
    # Play a game using MCTS until game will be done or max_moves will be reached
    while not game.done and len(game.action_history) < config['max_moves']:
        root = Node(0)
        
        # Create MinMaxStats Object to normalize values
        min_max_stats = MinMaxStats(config['min_value'], config['max_value'])
        
        # Expand the current root node
        curr_state = game.curr_state
        value = mcts.expand_root(root, list(range(config['action_space_size'])),
                            network, curr_state)
        mcts.backpropagate([root], value, config['discount'], min_max_stats)
        mcts.add_exploration_noise(config, root)

        # Run MCTS
        mcts.run_mcts(config, root, network, min_max_stats)

        # Select an action to take
        action = mcts.select_action(config, root)

        # Take an action and store tree search statistics
        game.take_action(action, env)
        game.store_search_statistics(root)
    print(f'Total reward for a train game: {sum(game.reward_history)}')
    return game


def test(config, network, env, test_rewards):
    """
    Test performance using trained networks
    """
    mcts = MCTS(config)
    returns = 0
    for _ in range(config['episodes_per_test']):
        # env.seed(1) # use for reproducibility of trajectories
        start_state, _ = env.reset()
        game = Game(config['action_space_size'], config['discount'], start_state)
        while not game.done and len(game.action_history) < config['max_moves']:
            min_max_stats = MinMaxStats(config['min_value'], config['max_value'])
            curr_state = game.curr_state
            root = Node(0)
            value = mcts.expand_root(root, list(range(config['action_space_size'])),
                                network, curr_state)
            mcts.backpropagate([root], value, config['discount'], min_max_stats)
            mcts.run_mcts(config, root, network, min_max_stats)
            action = mcts.select_action(config, root, test=True) # argmax action selection
            game.take_action(action, env)
        total_reward = sum(game.reward_history)
        print(f'Total reward for a test game: {total_reward}')
        test_rewards.add_reward(total_reward)
        returns += total_reward
    return returns / config['episodes_per_test']


config = {
          # Simulation and environment Config
          'action_space_size': 2, # number of action
          'state_shape': 4,
          'games_per_epoch': 20,
          'num_epochs': 25,
          'train_per_epoch': 30,
          'episodes_per_test': 10,
          'cartpole_stop_reward': 200,

          'visit_softmax_temperature_fn': 1,
          'max_moves': 200,
          'num_simulations': 50,
          'discount': 0.997,
          'min_value': 0,
          'max_value': 200,

          # Root prior exploration noise.
          'root_dirichlet_alpha': 0.1,
          'root_exploration_fraction': 0.25,

          # UCB parameters
          'pb_c_base': 19652,
          'pb_c_init': 1.25,

          # Model fitting config
          'embedding_size': 4,
          'hidden_neurons': 48,
          'buffer_size': 200,
          'batch_size': 512,
          'num_unroll_steps': 5,
          'td_steps': 10,
          'lr_init': 0.01,
          }


SEED = 0
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
value_support_size = math.ceil(math.sqrt(config['max_value'])) + 1


# Set seeds for reproducibility
set_seeds()

# Create the cartpole network
rep_net = RepresentationNetwork(input_size=config['state_shape'], hidden_neurons=config['hidden_neurons'], embedding_size=config['embedding_size']) # representation function
val_net = ValueNetwork(input_size=config['state_shape'], hidden_neurons=config['hidden_neurons'], value_support_size=value_support_size) # prediction function
pol_net = PolicyNetwork(input_size=config['state_shape'], hidden_neurons=config['hidden_neurons'], action_size=config['action_space_size']) # prediction function
dyn_net = DynamicNetwork(input_size=config['state_shape']+config['action_space_size'], hidden_neurons=config['hidden_neurons'], embedding_size=config['embedding_size']) # dynamics function
rew_net = RewardNetwork(input_size=config['state_shape']+config['action_space_size'], hidden_neurons=config['hidden_neurons']) # from dynamics function
network = Networks(rep_net, val_net, pol_net, dyn_net, rew_net, max_value=config['max_value'])

# action_size = 2
# state_shape = 4
# embedding_size = 4
# hidden_neurons = 48
# rep_net = RepresentationNetwork(input_size=state_shape, hidden_neurons=hidden_neurons, embedding_size=embedding_size) # representation function
# val_net = ValueNetwork(input_size=4, hidden_neurons=hidden_neurons, value_support_size=value_support_size) # prediction function
# pol_net = PolicyNetwork(input_size=4, hidden_neurons=hidden_neurons, action_size=action_size) # prediction function
# dyn_net = DynamicNetwork(input_size=6, hidden_neurons=hidden_neurons, embedding_size=embedding_size) # dynamics function
# rew_net = RewardNetwork(input_size=6, hidden_neurons=hidden_neurons) # from dynamics function
# network = Networks(rep_net, val_net, pol_net, dyn_net, rew_net, max_value=200)

# Create environment
env = gym.make('CartPole-v0')

# Create buffer to store games
replay_buffer = ReplayBuffer(config)
self_play(env, config, replay_buffer, network)

