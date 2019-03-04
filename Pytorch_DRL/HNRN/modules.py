import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import deque
import random
import pickle
from hmmlearn.hmm import GaussianHMM
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
        cdx, cdy, ndx, ndy, a_x, a_z, r, lc, ln = zip(*batch)

        s   = Variable(torch.FloatTensor(np.transpose([cdx, cdy])))
        s_n = Variable(torch.FloatTensor(np.transpose([ndx, ndy])))
        l   = Variable(torch.FloatTensor(lc))
        l_n = Variable(torch.FloatTensor(ln))
        a   = Variable(torch.FloatTensor(np.transpose([a_x, a_z])))
        r   = Variable(torch.FloatTensor(r))
        
        return s, s_n, a, r, l, l_n

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
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        self.X += self.theta * (self.mu-self.X) + self.sigma*np.random.randn(self.action_dim)
        return self.X


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

class Actor_Collision_Avoidance(nn.Module):
    def __init__(self, sensor_dim, action_dim):
        super(Actor_Collision_Avoidance, self).__init__()
        self.sensor_model = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Tanh())
        for i in [0, 2, 4, 6]:
            nn.init.xavier_uniform(self.sensor_model[i].weight.data)

    def forward(self, sensor):
        action_output = self.sensor_model(sensor)
        return action_output


# Critic is more complex than actor, which consists of two branches
# One is for state, the other is for action
class Critic_Collision_Avoidance(nn.Module):
    def __init__(self, sensor_dim, action_dim):
        super(Critic_Collision_Avoidance, self).__init__()
        self.sensor_model = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU())
        for i in [0, 2, 4]:
            nn.init.xavier_uniform(self.sensor_model[i].weight.data)

        self.state_model = nn.Sequential(
            nn.Linear(16+action_dim, 16), 
            nn.ReLU(),
            nn.Linear(16, 1))
        for i in [0, 2]:
            nn.init.xavier_uniform(self.state_model[i].weight.data)

    def forward(self, sensor, action):
        sensor_output = self.sensor_model(sensor)
        combine_state = torch.cat([sensor_output, action], 1)
        score_output = self.state_model(combine_state)

        return score_output


class Evaluation_Net():
    def __init__(self, n_state):
        print('[*] Building Evaluation Net...')
        self.n_state = n_state
        self.hmm = GaussianHMM(n_components=n_state, covariance_type="diag", n_iter=1000, tol=0.01)
        self.rewards = []
        self.map_index = []
        self.hidden_states = []

    def train_HMM(self, X, lengths, rewards):
        print('[*] Start training HMM...')
        self.rewards = rewards
        self.hmm = self.hmm.fit(X, lengths)

        # sort the state using reward
        self.sort_state_by_reward(X, lengths, rewards)

    def predict_state(self, X):
        raw_state = self.hmm.predict(X)
        return np.array([self.map_index.tolist().index(raw_state)])

    def print_monitor(self):
        print(self.hmm.monitor_)
    
    # output states' number of HMM is random, so a sort operatoin is needed
    def sort_state_by_reward(self, X, lengths, rewards):
        self.hidden_states = self.hmm.predict(X)
        map_from_state_to_reward = np.zeros(self.n_state)
        start_ = 0
        end_ = lengths[0]
        for i in range(len(lengths)):
            one_sequence = self.hidden_states[start_:end_]
            for j in range(len(one_sequence)):
                map_from_state_to_reward[one_sequence[j]] += rewards[i][j]

            if i+1 == len(lengths):
                break
            start_ = end_
            end_ += lengths[i+1]

        for i in range(self.n_state):
            if map_from_state_to_reward[i] != 0:
                map_from_state_to_reward[i] /= str(self.hidden_states.tolist()).count(str(i))

        print(map_from_state_to_reward)
        self.map_index = np.argsort(map_from_state_to_reward)

    def save_parameters(self):
        dbfile = open('models/HMM/hmm', 'w')
        pickle.dump(self.hmm, dbfile)
        dbfile.close()
        dbfile = open('models/HMM/map_index', 'w')
        pickle.dump(self.map_index, dbfile)
        dbfile.close()

    def load_parameters(self):
        dbfile = open('models/HMM/hmm', 'r')
        self.hmm = pickle.load(dbfile)
        dbfile.close()
        dbfile = open('models/HMM/map_index', 'r')
        self.map_index = pickle.load(dbfile)
        dbfile.close()
    
    def save_training_data_and_states(self, X):
        dbfile = open('models/HMM/training_data', 'w')
        pickle.dump(X, dbfile)
        dbfile.close()

        dbfile = open('models/HMM/hidden_states', 'w')
        pickle.dump(self.hidden_states, dbfile)
        dbfile.close()


class Differential_Driver():
    def __init__(self):
        print('[*] Building Differential Driver...')
    
    def run(self, x, y):
        return [[x, y]]
