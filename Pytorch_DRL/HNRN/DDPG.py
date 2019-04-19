import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# critic loss function
L2_loss_func = nn.MSELoss(reduction='mean')

import numpy as np
from collections import deque
import random
import pickle
import math

import utils
import modules

# Hierarchical Navigation Reinforcement Network
class DDPG(nn.Module):
    def __init__(self, train_type, max_buffer, state_dim, sensor_dim, target_dim, action_dim, mu, theta, sigma, actor_lr, critic_lr, batch_size, gamma, tau, hmm_state):
        super(DDPG, self).__init__()

        self.train_type = train_type
        self.state_dim = state_dim
        self.sensor_dim = sensor_dim
        self.target_dim = target_dim
        self.hmm_state = hmm_state

        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma

        # actor target driven
        self.actor_td = modules.Actor_Target_Driven(state_dim, action_dim)
        self.actor_td_target = modules.Actor_Target_Driven(state_dim, action_dim)
        self.actor_td_optimizer = optim.Adam(self.actor_td.parameters(), actor_lr)

        # critic target driven
        self.critic_td = modules.Critic_Target_Driven(state_dim, action_dim)
        self.critic_td_target = modules.Critic_Target_Driven(state_dim, action_dim)
        self.critic_td_optimizer = optim.Adam(self.critic_td.parameters(), critic_lr)

        # actor collision avoidance 
        self.actor_ca = modules.Actor_Collision_Avoidance(sensor_dim, target_dim, action_dim)
        self.actor_ca_target = modules.Actor_Collision_Avoidance(sensor_dim, target_dim, action_dim)
        self.actor_ca_optimizer = optim.Adam(self.actor_ca.parameters(), actor_lr)
        # critic collision avoidance 
        self.critic_ca = modules.Critic_Collision_Avoidance(sensor_dim, target_dim, action_dim)
        self.critic_ca_target = modules.Critic_Collision_Avoidance(sensor_dim, target_dim, action_dim)
        self.critic_ca_optimizer = optim.Adam(self.critic_ca.parameters(), critic_lr)

        self.buffer = modules.ReplayBuffer(max_buffer)
        self.noise = modules.OrnsteinUhlenbeckNoise(action_dim, mu, theta, sigma)

        # evaluation net
#        self.evaluation_net = modules.Evaluation_Net(n_state=hmm_state)

        # differential driver
#        self.differential_driver = modules.Differential_Driver()

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

    def sample_action(self, current_state, explore=True):

        # generate sensor data
        array_laser = utils.remapping_laser_data(current_state.laserScan)
        sensor = Variable(torch.FloatTensor(np.reshape(array_laser, (1, self.sensor_dim))))

        # generate target data
        target_polar = utils.target_transform(current_state)
        
        target = Variable(torch.FloatTensor(np.reshape(target_polar, (1, self.target_dim))))

        # generate action
        action = self.actor_ca(sensor=sensor, target=target).cpu().data.numpy()

        if explore and random.uniform(0,1) > 0.7 and self.train_type is not 3:
            action = action + self.noise.sample()
            #action[0][0] = random.uniform(-1,1)
            #action[0][1] = random.uniform(-1,1)

        # constrain the action
        action[0][0] = utils.constrain_actions(action[0][0], 1)
        action[0][1] = utils.constrain_actions(action[0][1], 1)

        return action[0][0], action[0][1]
        #return math.cos(target_polar[1]), math.sin(target_polar[1])

    def navigation(self, current_state):
        # generate sensor data
        array_laser = utils.remapping_laser_data(current_state.laserScan)
        sensor = Variable(torch.FloatTensor(np.reshape(array_laser, (1, self.sensor_dim))))

        # generate target data
        target_polar = utils.target_transform(current_state)
        target = Variable(torch.FloatTensor(np.reshape(target_polar, (1, self.target_dim))))
        
        # generate action
        target_driven_action = self.differential_driver.run(x=current_state.desired_x, y=current_state.desired_y)
        collision_avoidance_action = self.actor_ca(sensor=sensor, target=target).cpu().data.numpy()
        predict_state = self.evaluation_net.predict_state(array_laser.reshape(1, -1))

        # genrate action based on hmm_state
        # the less the state is, the dangerous the situation is
        # final_action = target_driven_action[0] + (self.hmm_state-predict_state)/float(self.hmm_state)*collision_avoidance_action[0]

        # Collision avoidance ratio
        ratio = min(float(torch.kthvalue(sensor,1)[0]) / (-3.5), 1)
#        print ratio
        final_action = []
        for i in range(2):
            final_action.append((1.0 - ratio) * target_driven_action[0][i] + ratio * collision_avoidance_action[0][i] )
#        print final_action

        # constrain the action
        final_action[0] = utils.constrain_actions(final_action[0], 1)
        final_action[1] = utils.constrain_actions(final_action[1], 1)
#        print final_action

        return final_action[0], final_action[1]

    def learn(self):
        if self.train_type == 1:
            return self.learn_target_driven()
        elif self.train_type == 2:
            #return self.learn_target_driven_supervised() 
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
        c_laser, c_target, action, reward, n_laser, n_target, terminal, desired_v = self.buffer.sample(self.batch_size)
        
        # change bool to tensor
        terminal =  Variable(torch.FloatTensor([1.0 * (i==False) for i in terminal]))

        self.critic_ca_optimizer.zero_grad()

        # The below 2 operations need to be detached since we only update eval and not targets
        a_n = self.actor_ca_target(sensor=n_laser, target=n_target).detach()
        q_value = self.critic_ca_target(sensor=n_laser, target=n_target, action=action).squeeze().detach()
        #print(reward + self.gamma*q_value)
        y_target = reward + self.gamma*q_value.mul(terminal)
        
        y_predicted = self.critic_ca(sensor=c_laser, target=c_target, action=action)

        # critic_loss = F.smooth_l1_loss(y_predicted, y_target)
        critic_loss = L2_loss_func(y_predicted, y_target)

        critic_loss.backward()
        self.critic_ca_optimizer.step()

        self.actor_ca_optimizer.zero_grad()

        a_pred = self.actor_ca(sensor=c_laser, target=c_target)
        q_pred = -self.critic_ca(sensor=c_laser, target=c_target, action=a_pred)

        actor_loss = q_pred.mean() 
        
        actor_loss.backward()
        self.actor_ca_optimizer.step()

        # update parameters
        self.update_targets()
        return critic_loss, actor_loss

    def learn_target_driven_supervised(self):
        c_laser, c_target, action, reward, n_laser, n_target, terminal, desired_v = self.buffer.sample(self.batch_size)

        # change bool to tensor
        terminal =  Variable(torch.FloatTensor([1.0 * (i==False) for i in terminal]))

        self.critic_ca_optimizer.zero_grad()

        # The below 2 operations need to be detached since we only update eval and not targets
        a_n = self.actor_ca_target(sensor=n_laser, target=n_target).detach()
        q_value = self.critic_ca_target(sensor=n_laser, target=n_target, action=action).squeeze().detach()
        #print(reward + self.gamma*q_value)
        y_target = reward + self.gamma*q_value.mul(terminal)
        
        y_predicted = self.critic_ca(sensor=c_laser, target=c_target, action=action)

        # critic_loss = F.smooth_l1_loss(y_predicted, y_target)
        critic_loss = L2_loss_func(y_predicted, y_target)

        critic_loss.backward()
        self.critic_ca_optimizer.step()

        # train actor
#        self.actor_ca_optimizer.zero_grad()
        a_pred = self.actor_ca(sensor=c_laser, target=c_target)

        actor_loss = L2_loss_func(a_pred, desired_v)

#        actor_loss.backward()
#        self.actor_ca_optimizer.step()

        # update parameters
        self.update_targets()
        return critic_loss, actor_loss

    def learn_hmm(self, X, lengths, rewards):
        self.evaluation_net.train_HMM(X, lengths, rewards)

    def save_hmm(self):
        self.evaluation_net.save_parameters()

    def load_hmm(self):
        self.evaluation_net.load_parameters()

    def save_hmm_data(self, X):
        self.evaluation_net.save_training_data_and_states(X)

    def save_models(self, actor_addr, critic_addr):
#        if self.train_type == 1:
#            torch.save(self.actor_td.state_dict(), 'models/best_td_actor.model')
#            torch.save(self.critic_td.state_dict(), 'models/best_td_critic.model')
#        elif self.train_type == 2:
#            torch.save(self.actor_ca.state_dict(), 'models/best_ca_actor.model')
#            torch.save(self.critic_ca.state_dict(), 'models/best_ca_critic.model')
        torch.save(self.actor_ca.state_dict(), actor_addr)
        torch.save(self.critic_ca.state_dict(), critic_addr)

    def load_models(self, actor_addr, critic_addr):
#        if self.train_type == 1:
#            self.actor_td.load_state_dict(torch.load('models/best_td_actor.model'))
#            self.critic_td.load_state_dict(torch.load('models/best_td_critic.model'))
#        elif self.train_type == 2:
#            self.actor_ca.load_state_dict(torch.load('models/best_ca_actor.model'))
#            self.critic_ca.load_state_dict(torch.load('models/best_ca_critic.model'))
        self.actor_ca.load_state_dict(torch.load(actor_addr))
        self.critic_ca.load_state_dict(torch.load(critic_addr))

    def save_buffer(self):
        self.buffer.save_buffer()
    
    def load_buffer(self):
        self.buffer.load_buffer()
