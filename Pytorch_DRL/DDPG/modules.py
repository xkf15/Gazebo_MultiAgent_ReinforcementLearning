import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import deque
import random
import pickle
import warnings
warnings.filterwarnings("ignore")


class ReplayBuffer:

    def __init__(self, max_buffer):
        self.buffer_size = max_buffer
        self.len = 0

        # Create buffers 
        self.buffer = deque(maxlen=self.buffer_size)

    def sample(self, count):
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        # pop out experience
        cs, ct, a, r, ns, nt, terminal, v = zip(*batch)

        c_target = Variable(torch.FloatTensor(ct))
        n_target = Variable(torch.FloatTensor(nt))
        c_laser = Variable(torch.FloatTensor(cs))
        n_laser = Variable(torch.FloatTensor(ns))
        action = Variable(torch.FloatTensor(a))
        reward = Variable(torch.FloatTensor(r))
        desired_v = Variable(torch.FloatTensor(v))
        
        return c_laser, c_target, action, reward, n_laser, n_target, terminal, desired_v

    def add(self, to_be_saved):
        self.len += 1
        if self.len > self.buffer_size:
            self.len = self.buffer_size
        self.buffer.append(to_be_saved)

    def save_buffer(self):
        dbfile = open('experiences/buffer', 'w')
        pickle.dump(self.buffer, dbfile)
        dbfile.close()

    def load_buffer(self):
        dbfile = open('experiences/buffer', 'r')
        self.buffer = pickle.load(dbfile)
        dbfile.close()


# the explore stratagy is not epsilon-greedy, but adding a noise to it
class OrnsteinUhlenbeckNoise():
    def __init__(self, action_dim, mu, theta, sigma):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_origin = sigma
        self.X = np.ones(self.action_dim) * self.mu
        self.decayStep = 0

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        self.X += self.theta * (self.mu-self.X) + self.sigma * np.random.randn(self.action_dim)
        return self.X
    
    def decay(self):
        self.sigma = self.sigma_origin * 10.0 / (self.decayStep + 10.0)
        self.decayStep += 1
        return self.sigma

class Actor_Target_Driven(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor_Target_Driven, self).__init__()
        self.state_model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh())
        for i in [0, 2, 4]:
            nn.init.xavier_uniform(self.state_model[i].weight.data)

    def forward(self, state):
        action_output = self.state_model(state)

        return action_output


class Critic_Target_Driven(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_Target_Driven, self).__init__()

        self.state_model = nn.Sequential(
            nn.Linear(state_dim+action_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
        for i in [0, 2, 4]:
            nn.init.xavier_uniform(self.state_model[i].weight.data)

    def forward(self, state, action):
        combine_state = torch.cat([state, action], 1)
        score_output = self.state_model(combine_state)

        return score_output

# Actor Model
class Actor_Collision_Avoidance(nn.Module):
    def __init__(self, sensor_dim, target_dim, action_dim):
        super(Actor_Collision_Avoidance, self).__init__()
        self.sensor_model = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU())
        for i in [0, 2]:
            nn.init.xavier_uniform(self.sensor_model[i].weight.data)
        
        self.target_model = nn.Sequential(
            nn.Linear(64 + target_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Tanh())
        for i in [0, 2]:
            nn.init.xavier_uniform(self.target_model[i].weight.data)
    
    # sensor 360 dim, target polar coordinate
    def forward(self, sensor, target):
        sensor_output = self.sensor_model(sensor)
        combine_state = torch.cat([sensor_output, target], 1)
        action_output = self.target_model(combine_state)

        return action_output


# Critic Model
class Critic_Collision_Avoidance(nn.Module):
    def __init__(self, sensor_dim, target_dim, action_dim):
        super(Critic_Collision_Avoidance, self).__init__()
        self.sensor_model = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU())
        for i in [0, 2]:
            nn.init.xavier_uniform(self.sensor_model[i].weight.data)
        
        self.target_model = nn.Sequential(
            nn.Linear(64 + target_dim, 16),
            nn.ReLU())
        for i in [0]:
            nn.init.xavier_uniform(self.target_model[i].weight.data)
        
        self.state_model = nn.Sequential(
            nn.Linear(16 + action_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1))
        for i in [0, 2]:
            nn.init.xavier_uniform(self.state_model[i].weight.data)

    def forward(self, sensor, target, action):
        sensor_output = self.sensor_model(sensor)
        combine_state = torch.cat([sensor_output, target], 1)
        state_output = self.target_model(combine_state)
        combine_action = torch.cat([state_output, action], 1)
        score_output = self.state_model(combine_action)

        return score_output


