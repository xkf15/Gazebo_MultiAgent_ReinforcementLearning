import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import deque
import random
import pickle

import utils


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
        cvx, cvy, cow, cox, coy, coz, nvx, nvy, now, nox, noy, noz, a_x, a_z, r, lc, ln = zip(*batch)

        s   = Variable(torch.FloatTensor(np.transpose([cvx, cvy, cow, cox, coy, coz])))
        s_n = Variable(torch.FloatTensor(np.transpose([nvx, nvy, now, nox, noy, noz])))
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
        self.X += self.theta * (self.mu - self.X) + self.sigma * np.random.randn(self.action_dim)
        return self.X


class Actor(nn.Module):
    def __init__(self, state_dim, sensor_dim, action_dim):
        super(Actor, self).__init__()
        self.sensor_model = nn.Sequential(
            nn.Linear(sensor_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU())
        for i in [0, 2, 4, 6]:
            nn.init.xavier_uniform(self.sensor_model[i].weight.data)

        self.state_model = nn.Sequential(
            nn.Linear(state_dim+32, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh())
        for i in [0, 2, 4]:
            nn.init.xavier_uniform(self.state_model[i].weight.data)

        self.action_layer_x = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid())
        self.action_layer_z = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh())
        nn.init.xavier_uniform(self.action_layer_x[0].weight.data)
        nn.init.xavier_uniform(self.action_layer_z[0].weight.data)

    def forward(self, state, sensor):
        sensor_output = self.sensor_model(sensor)
        combine_state = torch.cat([sensor_output, state], 1)

        '''
        action_output_x = self.action_layer_x(self.state_model(combine_state))
        action_output_z = self.action_layer_z(self.state_model(combine_state))
        action_output = torch.cat([action_output_x, action_output_z], 1)
        '''

        action_output = self.state_model(combine_state)

        return action_output

# Critic is more complex than actor, which consists of two branches
# One is for state, the other is for action
class Critic(nn.Module):
    def __init__(self, state_dim, sensor_dim, action_dim):
        super(Critic, self).__init__()
        self.sensor_model = nn.Sequential(
            nn.Linear(sensor_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU())
        for i in [0, 2, 4, 6]:
            nn.init.xavier_uniform(self.sensor_model[i].weight.data)

        self.state_model = nn.Sequential(
            nn.Linear(state_dim+32+action_dim, 32), 
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
        for i in [0, 2, 4]:
            nn.init.xavier_uniform(self.state_model[i].weight.data)

    def forward(self, state, sensor, action):
        sensor_output = self.sensor_model(sensor)
        combine_state = torch.cat([sensor_output, state], 1)
        combine_state = torch.cat([combine_state, action], 1)
        score_output = self.state_model(combine_state)
        return score_output


class DDPG(nn.Module):
    def __init__(self, max_buffer, state_dim, sensor_dim, action_dim, mu, theta, sigma, actor_lr, critic_lr, batch_size, gamma, tau):
        super(DDPG, self).__init__()

        self.state_dim = state_dim
        self.sensor_dim = sensor_dim

        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma

        self.actor = Actor(state_dim, sensor_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), actor_lr)
        self.actor_target = Actor(state_dim, sensor_dim, action_dim)

        self.critic = Critic(state_dim, sensor_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), critic_lr)
        self.critic_target = Critic(state_dim, sensor_dim, action_dim)

        self.buffer = ReplayBuffer(max_buffer)
        self.noise = OrnsteinUhlenbeckNoise(action_dim, mu, theta, sigma)

    def update_targets(self):
        for actor, actor_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            actor_target.data.copy_(self.tau*actor.data + (1-self.tau)*actor_target.data)

        for critic, critic_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            critic_target.data.copy_(self.tau*critic.data + (1-self.tau)*critic_target.data)

    def copy_weights(self):
        for actor, actor_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            actor_target.data.copy_(actor.data)

        for critic, critic_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            critic_target.data.copy_(critic.data)

    def sample_action(self, current_state, explore=True):

        # generate states
        cvx, cvy = utils.vector_normalization(current_state.target_x, current_state.target_y, current_state.current_x, current_state.current_y)

        # reshape is needed here, because single sample and batch sample should be both 2-dim
        state = Variable(torch.FloatTensor(np.transpose(np.reshape([cvx, cvy,
                                                                    current_state.orientation_w,current_state.orientation_x,
                                                                    current_state.orientation_y,current_state.orientation_z], 
                                                                    (self.state_dim, 1)))))
        '''
        state = Variable(torch.FloatTensor(np.transpose(np.reshape([current_state.current_x, current_state.current_y,
                                                                    current_state.target_x, current_state.target_y,
                                                                    current_state.orientation_w,current_state.orientation_x,
                                                                    current_state.orientation_y,current_state.orientation_z], 
                                                                    (self.state_dim, 1)))))
        '''

        # generate sensor data
        array_laser = np.array(current_state.laserScan)
        where_inf = np.isinf(array_laser)
        array_laser[where_inf] = 3.5
        sensor = Variable(torch.FloatTensor(np.reshape(array_laser, (1, self.sensor_dim))))

        # generate action
        action = self.actor(state=state, sensor=sensor).cpu().data.numpy()

        # amplify the linear speed
        #action[0][0] = 2*action[0][0]

        if explore:
            if random.uniform(0,1) > 0.7:
                #action = action + self.noise.sample()
                action[0][0] = random.uniform(-1,1)
                action[0][1] = random.uniform(-1,1)

        # constrain the action
        action[0][0] = utils.constrain_actions(action[0][0], 1)
        action[0][1] = utils.constrain_actions(action[0][1], 1)

        return action[0][0], action[0][1]

    def learn(self):
        s, s_n, a, r, l, l_n = self.buffer.sample(self.batch_size)

        # The below 2 operations need to be detached since we only update eval and not targets

        a_n = self.actor_target(s_n, l_n).detach()
        q_value = self.critic_target(s_n, l_n, a_n).squeeze().detach()
        y_target = r + self.gamma*q_value
        y_predicted = self.critic(s, l, a)

        critic_loss = F.smooth_l1_loss(y_predicted, y_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        a_pred = self.actor(s, l)
        q_pred = -self.critic(s, l, a_pred)

        actor_loss = q_pred.mean()  # because we want to maximize q_pred
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update parameters
        self.update_targets()

        return critic_loss, actor_loss

    def save_models(self):
        torch.save(self.actor_target.state_dict(), 'models/best_actor.model')
        torch.save(self.critic_target.state_dict(), 'models/best_critic.model')
    
    def load_models(self):
        self.actor_target.load_state_dict(torch.load('models/best_actor.model'))
        self.critic_target.load_state_dict(torch.load('models/best_critic.model'))

    def save_buffer(self):
        self.buffer.save_buffer()
    
    def load_buffer(self):
        self.buffer.load_buffer()
