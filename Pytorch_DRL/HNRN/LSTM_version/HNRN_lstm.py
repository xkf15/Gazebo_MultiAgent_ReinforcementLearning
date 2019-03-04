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
import modules

# Hierarchical Navigation Reinforcement Network
class HNRN(nn.Module):
    def __init__(self, train_type, max_buffer, state_dim, sensor_dim, action_dim, mu, theta, sigma, actor_lr, critic_lr, batch_size, gamma, tau, lstm_hidden_size):
        super(HNRN, self).__init__()

        self.train_type = train_type
        self.state_dim = state_dim
        self.sensor_dim = sensor_dim

        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.lstm_hidden_size = lstm_hidden_size

        # actor target driven
        self.actor_td = modules.Actor_Target_Driven(state_dim, action_dim)
        self.actor_td_target = modules.Actor_Target_Driven(state_dim, action_dim)
        self.actor_td_optimizer = optim.Adam(self.actor_td.parameters(), actor_lr)

        # critic target driven
        self.critic_td = modules.Critic_Target_Driven(state_dim, action_dim)
        self.critic_td_target = modules.Critic_Target_Driven(state_dim, action_dim)
        self.critic_td_optimizer = optim.Adam(self.critic_td.parameters(), critic_lr)

        # actor collision avoidance 
        self.actor_ca = modules.Actor_Collision_Avoidance(sensor_dim, action_dim)
        self.actor_ca_target = modules.Actor_Collision_Avoidance(sensor_dim, action_dim)
        self.actor_ca_optimizer = optim.Adam(self.actor_ca.parameters(), actor_lr)
        # critic collision avoidance 
        self.critic_ca = modules.Critic_Collision_Avoidance(sensor_dim, action_dim)
        self.critic_ca_target = modules.Critic_Collision_Avoidance(sensor_dim, action_dim)
        self.critic_ca_optimizer = optim.Adam(self.critic_ca.parameters(), critic_lr)

        self.buffer = modules.ReplayBuffer(max_buffer)
        self.noise = modules.OrnsteinUhlenbeckNoise(action_dim, mu, theta, sigma)

        # differential driver
        self.differential_driver = modules.Differential_Driver()

    def update_targets(self):
        if self.train_type == 1:
            for actor_td, actor_td_target in zip(self.actor_td.parameters(), self.actor_td_target.parameters()):
                actor_td_target.data.copy_(self.tau*actor_td.data + (1-self.tau)*actor_td_target.data)
            for critic_td, critic_td_target in zip(self.critic_td.parameters(), self.critic_td_target.parameters()):
                critic_td_target.data.copy_(self.tau*critic_td.data + (1-self.tau)*critic_td_target.data)
        elif self.train_type == 2:
            for actor_ca, actor_ca_target in zip(self.actor_ca.parameters(), self.actor_ca_target.parameters()):
                actor_ca_target.data.copy_(self.tau*actor_ca.data + (1-self.tau)*actor_ca_target.data)
            for critic_ca, critic_ca_target in zip(self.critic_ca.parameters(), self.critic_ca_target.parameters()):
                critic_ca_target.data.copy_(self.tau*critic_ca.data + (1-self.tau)*critic_ca_target.data)

    def copy_weights(self):
        if self.train_type == 1:
            for actor_td, actor_td_target in zip(self.actor_td.parameters(), self.actor_td_target.parameters()):
                actor_td_target.data.copy_(actor_td.data)
            for critic_td, critic_td_target in zip(self.critic_td.parameters(), self.critic_td_target.parameters()):
                critic_td_target.data.copy_(critic_td.data)
        elif self.train_type == 2:
            for actor_ca, actor_ca_target in zip(self.actor_ca.parameters(), self.actor_ca_target.parameters()):
                actor_ca_target.data.copy_(actor_ca.data)
            for critic_ca, critic_ca_target in zip(self.critic_ca.parameters(), self.critic_ca_target.parameters()):
                critic_ca_target.data.copy_(critic_ca.data)

    def sample_action(self, current_state, laser_data, explore=True):

        # reshape is needed here, because single sample and batch sample should be both 2-dim
        state = Variable(torch.FloatTensor(np.transpose(np.reshape([current_state.desired_x, current_state.desired_y], (self.state_dim, 1)))))
        # generate sensor data
        array_laser_1 = utils.remapping_laser_data(laser_data[0])
        array_laser_2 = utils.remapping_laser_data(laser_data[1])
        array_laser_3 = utils.remapping_laser_data(current_state.laserScan)

        sensor_1 = Variable(torch.FloatTensor(np.reshape(array_laser_1, (1, self.sensor_dim))))
        sensor_2 = Variable(torch.FloatTensor(np.reshape(array_laser_2, (1, self.sensor_dim))))
        sensor_3 = Variable(torch.FloatTensor(np.reshape(array_laser_3, (1, self.sensor_dim))))
        sensor = torch.cat([sensor_1, sensor_2, sensor_3], 1)

        # generate action
        if self.train_type == 1:
            action = self.actor_td(state=state).cpu().data.numpy()
        elif self.train_type == 2:
            action = self.actor_ca(sensor=sensor).cpu().data.numpy()
        else:
            action = self.differential_driver.run(x=current_state.desired_x, y=current_state.desired_y)

        if explore and random.uniform(0,1) > 0.7 and self.train_type is not 3:
            action = action + self.noise.sample()
            #action[0][0] = random.uniform(-1,1)
            #action[0][1] = random.uniform(-1,1)

        # constrain the action
        action[0][0] = utils.constrain_actions(action[0][0], 1)
        action[0][1] = utils.constrain_actions(action[0][1], 1)

        return action[0][0], action[0][1]

    def learn(self):
        if self.train_type == 1:
            return self.learn_target_driven()
        elif self.train_type == 2:
            return self.learn_collision_avoidance()

    def learn_target_driven(self):
        s, s_n, a, r, _, _ = self.buffer.sample(self.batch_size)

        # The below 2 operations need to be detached since we only update eval and not targets
        a_n = self.actor_td_target(state=s_n).detach()
        q_value = self.critic_td_target(state=s_n, action=a_n).squeeze().detach()
        y_target = r + self.gamma*q_value
        y_predicted = self.critic_td(state=s, action=a)

        critic_loss = F.smooth_l1_loss(y_predicted, y_target)
        self.critic_td_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_td_optimizer.step()

        a_pred = self.actor_td(state=s)
        q_pred = -self.critic_td(state=s, action=a_pred)

        actor_loss = q_pred.mean()  # because we want to maximize q_pred
        self.actor_td_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_td_optimizer.step()

        # update parameters
        self.update_targets()

        return critic_loss, actor_loss
    
    def learn_collision_avoidance(self):
        _, _, a, r, l, l_n = self.buffer.sample(self.batch_size)

        # The below 2 operations need to be detached since we only update eval and not targets
        a_n = self.actor_ca_target(sensor=l_n).detach()
        q_value = self.critic_ca_target(sensor=l_n, action=a_n).squeeze().detach()
        y_target = r + self.gamma*q_value
        y_predicted = self.critic_ca(sensor=l, action=a)

        critic_loss = F.smooth_l1_loss(y_predicted, y_target)
        self.critic_ca_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_ca_optimizer.step()

        a_pred = self.actor_ca(sensor=l)
        q_pred = -self.critic_ca(sensor=l, action=a_pred)

        actor_loss = q_pred.mean() 
        self.actor_ca_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_ca_optimizer.step()

        # update parameters
        self.update_targets()
        return critic_loss, actor_loss

    def save_models(self):
        if self.train_type == 1:
            torch.save(self.actor_td.state_dict(), 'models/best_td_actor.model')
            torch.save(self.critic_td.state_dict(), 'models/best_td_critic.model')
        elif self.train_type == 2:
            torch.save(self.actor_ca.state_dict(), 'models/best_ca_actor.model')
            torch.save(self.critic_ca.state_dict(), 'models/best_ca_critic.model')

    def load_models(self):
        if self.train_type == 1:
            self.actor_td.load_state_dict(torch.load('models/best_td_actor.model'))
            self.critic_td.load_state_dict(torch.load('models/best_td_critic.model'))
        elif self.train_type == 2:
            self.actor_ca.load_state_dict(torch.load('models/best_ca_actor.model'))
            self.critic_ca.load_state_dict(torch.load('models/best_ca_critic.model'))

    def save_buffer(self):
        self.buffer.save_buffer()
    
    def load_buffer(self):
        self.buffer.load_buffer()
