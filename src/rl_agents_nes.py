import numpy as np
import cv2
import random
import gym
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from collections import deque
import os
import pathlib
from gym import spaces
from rl_models_nes import DQN, DuelingDQN

class DQNAgent:
    def __init__(self, num_states, num_actions, capacity, batch_size, gamma, lr, replay_init, loss_function, weights_dir = None):

        self.state_space = num_states
        self.action_space = num_actions
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.net = DQN( self.state_space, self.action_space).to(self.device)
        self.max_memory_size = capacity
        self.memory_batch_size = batch_size
        self.replay_initialization = replay_init

        # Create replay memory
        self.states1_mem = torch.zeros(self.max_memory_size, *self.state_space)
        self.actions_mem = torch.zeros(self.max_memory_size, 1)
        self.rewards_mem = torch.zeros(self.max_memory_size, 1)
        self.states2_mem = torch.zeros(self.max_memory_size, *self.state_space)
        self.dones_mem = torch.zeros(self.max_memory_size, 1)
        self.last_cell = 0
        self.current_size = 0

        #Load a pre-trained network
        if weights_dir == None:
            self.dir_weights = ""       
        else:
            self.dir_weights = weights_dir
            self.net.load_state_dict(torch.load(os.path.join(self.dir_weights, "DQN.pt"), map_location=torch.device(self.device)))

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        # Learning parameters
        self.gamma = gamma
        if loss_function == "Huber":
            self.l1 = nn.SmoothL1Loss().to(self.device) #Huber loss
        else:
            self.l1 = nn.MSELoss().to(self.device) #MSE loss

    def memorize(self, state, action, reward, new_state, done):
        '''Store the experiences in a buffer to use later'''
        self.states1_mem[self.last_cell] = state.float()
        self.actions_mem[self.last_cell] = action.float()
        self.rewards_mem[self.last_cell] = reward.float()
        self.states2_mem[self.last_cell] = new_state.float()
        self.dones_mem[self.last_cell] = done.float()
        self.last_cell = (self.last_cell + 1) % self.max_memory_size  # FIFO tensor
        self.current_size = min(self.current_size + 1, self.max_memory_size)
    
    def replay(self):
        '''Randomly sample batch size experiences'''
        idx = random.choices(range(self.current_size), k=self.memory_batch_size)
        state = self.states1_mem[idx]
        action = self.actions_mem[idx]
        reward = self.rewards_mem[idx]
        new_state = self.states2_mem[idx]
        done = self.dones_mem[idx]      
        return state, action, reward, new_state, done
    
    def act(self, state, epsilon):
        '''Choose an action according to epsilong-greedy policy'''
        if random.random() < epsilon:  
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
            return torch.argmax(self.net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
    
    def experience_replay(self):
        '''Use the Q-update equations to update the network weights'''

        #check if the memory is quite full to update the network weights 
        if self.memory_batch_size > self.max_memory_size:
            return
        
        if self.current_size <= self.replay_initialization:
            return 
    
        # Sample a batch of experiences
        state, action, reward, next_state, done = self.replay()
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        target = reward + torch.mul((self.gamma * self.net(next_state).max(1).values.unsqueeze(1)), 1 - done)      
        current = self.net(state).gather(1, action.long())           
        
        loss = self.l1(current, target)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error

        return loss.item()

class DuelingAgent:
    def __init__(self, num_states, num_actions, memory_size, batch_size, gamma, learning_rate, replay_init, is_double_dqn = False, loss_function = "Huber", weights_dir = None):
        
        self.state_space = num_states
        self.action_space = num_actions
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.double_enabled = is_double_dqn
        
        self.step = 0

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

        self.max_memory_size = memory_size
        self.memory_batch_size = batch_size
        self.replay_initialization = replay_init

        # Create memory and load pre-trained model
        self.states1_mem = torch.zeros(self.max_memory_size, *self.state_space)
        self.actions_mem = torch.zeros(self.max_memory_size, 1)
        self.rewards_mem = torch.zeros(self.max_memory_size, 1)
        self.states2_mem = torch.zeros(self.max_memory_size, *self.state_space)
        self.dones_mem = torch.zeros(self.max_memory_size, 1)
        self.last_cell = 0
        self.current_size = 0
        
        # Learning parameters
        self.gamma = gamma
        if loss_function == "Huber":
            self.l1 = nn.SmoothL1Loss().to(self.device) #Huber loss
        else:
            self.l1 = nn.MSELoss().to(self.device) #MSE loss

    def memorize(self, state, action, reward, new_state, done):
        """Store the experiences in a buffer to use later"""
        self.states1_mem[self.last_cell] = state.float()
        self.actions_mem[self.last_cell] = action.float()
        self.rewards_mem[self.last_cell] = reward.float()
        self.states2_mem[self.last_cell] = new_state.float()
        self.dones_mem[self.last_cell] = done.float()
        self.last_cell = (self.last_cell + 1) % self.max_memory_size  # FIFO tensor
        self.current_size = min(self.current_size + 1, self.max_memory_size)
    
    def replay(self):
        '''Randomly sample 'batch size' experiences'''
        idx = random.choices(range(self.current_size), k=self.memory_batch_size)
        state = self.states1_mem[idx]
        action = self.actions_mem[idx]
        reward = self.rewards_mem[idx]
        new_state = self.states2_mem[idx]
        done = self.dones_mem[idx]      
        return state, action, reward, new_state, done
    
    def act(self, state, epsilon):
        '''Choose an action according to epsilong-greedy policy'''
        self.step += 1
        if random.random() < epsilon:  
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
            if self.double_enabled:
                q_value = self.policy_net(state.to(self.device))
            else:
                q_value = self.net(state.to(self.device))
            
            action = torch.argmax(q_value).unsqueeze(0).unsqueeze(0).cpu()      
            return action
    
    def experience_replay(self):
        '''Use the Q-update equations to update the network weights'''

        if self.double_enabled and self.step % 10000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.memory_batch_size > self.max_memory_size:
            return
        
        if self.current_size <= self.replay_initialization:
            return 
    
        # Sample a batch of experiences
        state, action, reward, next_state, done = self.replay()
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        self.optimizer.zero_grad()
          
        if self.double_enabled:            
            # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
            target = reward + torch.mul((self.gamma * self.target_net(next_state).max(1).values.unsqueeze(1)), 1 - done)      
            current = self.policy_net(state).gather(1, action.long())  
        else:
            # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
            target = reward + torch.mul((self.gamma * self.net(next_state).max(1).values.unsqueeze(1)), 1 - done)      
            current = self.net(state).gather(1, action.long())       
        
        loss = self.l1(current, target)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error

        return loss.item()
