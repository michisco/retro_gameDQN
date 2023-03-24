import numpy as np
import cv2
import random
import gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from collections import deque
import os
import pathlib
from gym import spaces
from rl_models import DQN, DuelingDQN, ReplayMemory

class DQNAgent:
    def __init__(self, num_states, num_actions, memory_size, replay_init, batch_size, gamma, learning_rate, loss_function = "Huber", weights_dir = None):        
        
        self.state_space = num_states
        self.action_space = num_actions
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.net = DQN(self.state_space, self.action_space).to(self.device)

        #Load a pre-trained network
        if weights_dir == None:
            self.dir_weights = ""
        else:
            self.dir_weights = weights_dir
            self.net.load_state_dict(torch.load(os.path.join(self.dir_weights, "DQN.pt"), map_location=torch.device(self.device)))
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.step = 0

        #Create replay memory
        self.memory_capacity = memory_size
        self.replay_buffer = ReplayMemory(self.memory_capacity)
        self.memory_sample_size = batch_size
        self.replay_initialization = replay_init

        #Set learning parameters
        self.gamma = gamma      
        if loss_function == "Huber":
            self.l1 = nn.SmoothL1Loss().to(self.device) #is closely related to Huber loss -> equivalent huber(x,y)/beta
        else:
            self.l1 = nn.MSELoss().to(self.device) #MSE loss

    def act(self, state, epsilon):
        '''Choose an action according to epsilong-greedy policy'''
        with torch.no_grad():
            self.step += 1
            if random.random() > epsilon:
                state   = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_value = self.net(state)
                action  = q_value.max(1)[1].item()
            else:
                action = random.randrange(self.action_space)
        return action
    
    def memorize(self, state, action, reward, new_state, done):
        self.replay_buffer.store(state, action, new_state, reward, done)
    
    def experience_replay(self):
        '''Use the Q-update equations to update the network weights'''

        #check if the memory is quite full to update the network weights 
        if self.memory_sample_size > self.replay_buffer.__len__():
            return
        
        if self.replay_buffer.__len__() <= self.replay_initialization:
            return
        
        # Sample a batch of experiences
        state, action, next_state, reward, done = self.replay_buffer.replay(self.memory_sample_size)
        state_batch    = torch.tensor(state).to(self.device)
        next_state_batch = torch.tensor(np.array(next_state), requires_grad=False).to(self.device)

        action_batch    = torch.LongTensor(action).to(self.device)
        reward_batch    = torch.FloatTensor(reward).to(self.device)
        done_batch       = torch.FloatTensor(done).to(self.device)
        
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
        self.optimizer.zero_grad()
        q_values      = self.net(state_batch)   
        current       = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        next_q_values = self.net(next_state_batch)
        next_q_value  = next_q_values.max(1)[0]
        target = reward_batch + torch.mul(self.gamma * next_q_value, (1. - done_batch))
        
        loss = self.l1(current, target.data)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error

        return loss.item()

class DuelingAgent:
    def __init__(self, num_states, num_actions, memory_size, replay_init, batch_size, gamma, learning_rate, is_double_dqn = False, loss_function = "Huber", weights_dir = None):

        self.state_space = num_states
        self.action_space = num_actions
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.double_enabled = is_double_dqn
        
        if self.double_enabled:
            self.policy_net = DuelingDQN(self.state_space, self.action_space).to(self.device)
            self.target_net = DuelingDQN(self.state_space, self.action_space).to(self.device)
            
            #Load the pre-trained networks
            if weights_dir == None:
                self.dir_weights = ""
            else:
                self.dir_weights = weights_dir
                self.policy_net.load_state_dict(torch.load(os.path.join(self.dir_weights, "DuelingDQN_policy.pt"), map_location=torch.device(self.device)))
                self.target_net.load_state_dict(torch.load(os.path.join(self.dir_weights, "DuelingDQN_target.pt"), map_location=torch.device(self.device)))
            
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        else:
            self.net = DuelingDQN(self.state_space, self.action_space).to(self.device)
            
            #Load a pre-trained network
            if weights_dir == None:
                self.dir_weights = ""
            else:
                self.dir_weights = weights_dir
                self.net.load_state_dict(torch.load(os.path.join(self.dir_weights, "DuelingDQN.pt"), map_location=torch.device(self.device)))
            
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self.step = 0

        #Create replay memory
        self.memory_capacity = memory_size
        self.replay_buffer = ReplayMemory(self.memory_capacity)
        self.memory_sample_size = batch_size
        self.replay_initialization = replay_init

        #Set learning parameters
        self.gamma = gamma      
        if loss_function == "Huber":
            self.l1 = nn.SmoothL1Loss().to(self.device) #Huber loss
        else:
            self.l1 = nn.MSELoss().to(self.device) #MSE loss

    def act(self, state, epsilon):
        '''Choose an action according to epsilong-greedy policy'''
        with torch.no_grad():
            self.step += 1
            if random.random() > epsilon:
                state   = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if self.double_enabled:
                    q_value = self.policy_net(state)
                else:
                    q_value = self.net(state)
                action  = q_value.max(1)[1].item()
            else:
                action = random.randrange(self.action_space)
        return action
    
    def memorize(self, state, action, reward, new_state, done):
        self.replay_buffer.store(state, action, new_state, reward, done)
    
    def experience_replay(self):
        '''Use the Q-update equations to update the network weights'''

        #if the agent is a Double DQN then update the target network every few frames
        if self.double_enabled and self.step % 10000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        #check if the memory is quite full to update the network weights 
        if self.memory_sample_size > self.replay_buffer.__len__():
            return
        
        if self.replay_buffer.__len__() <= self.replay_initialization:
            return
        
        # Sample a batch of experiences
        state, action, next_state, reward, done = self.replay_buffer.replay(self.memory_sample_size)
        state_batch    = torch.tensor(state).to(self.device)
        next_state_batch = torch.tensor(np.array(next_state), requires_grad=False).to(self.device)

        action_batch    = torch.LongTensor(action).to(self.device)
        reward_batch    = torch.FloatTensor(reward).to(self.device)
        done_batch       = torch.FloatTensor(done).to(self.device)
        
        self.optimizer.zero_grad()
        # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
        if self.double_enabled:
            q_values      = self.policy_net(state_batch)   
            current       = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            next_q_values = self.target_net(next_state_batch)
            next_q_value  = next_q_values.max(1)[0]
            target = reward_batch + torch.mul(self.gamma * next_q_value, (1. - done_batch))
        else:
            # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
            q_values      = self.net(state_batch)   
            current       = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            next_q_values = self.net(next_state_batch)
            next_q_value  = next_q_values.max(1)[0]
            target = reward_batch + torch.mul(self.gamma * next_q_value, (1. - done_batch))
        
        loss = self.l1(current, target.data)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error

        return loss.item()