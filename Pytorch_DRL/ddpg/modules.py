import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import utils


class ReplayBuffer:

    def __init__(self, max_buffer):
        self.buffer_size = max_buffer
        self.len = 0

        # Create buffers for (s_t, a_t, r_t, s_t+1)
        self.buffer = deque(maxlen=self.buffer_size)

    def sample(self, count):
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        cx, cy, cow, cox, coy, coz, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz, a_x, a_z, r = zip(*batch)
        s   = Variable(torch.FloatTensor(np.transpose((cx,cy,cow,cox,coy,coz,tx,ty,tow,tox,toy,toz))))
        s_n = Variable(torch.FloatTensor(np.transpose((nx,ny,now,nox,noy,noz,tx,ty,tow,tox,toy,toz))))
        a = Variable(torch.FloatTensor(np.transpose((a_x,a_z))))
        r = Variable(torch.FloatTensor(r))

        return s, s_n, a, r

    def add(self, to_be_saved):
        self.len += 1
        if self.len > self.buffer_size:
            self.len = self.buffer_size
        self.buffer.append(to_be_saved)

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
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh())

        for i in [0, 2, 4]:
            nn.init.xavier_uniform(self.model[i].weight.data)

        self.model[-2].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        return self.model(state)

# Critic is more complex than actor, which consists of two branches
# One is for state, the other is for action
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.transform_state = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        nn.init.xavier_uniform(self.transform_state[0].weight.data)

        self.transform_action = nn.Sequential(nn.Linear(action_dim, 128), nn.ReLU())
        nn.init.xavier_uniform(self.transform_action[0].weight.data)

        # the output dimension is 1, which is for scoring
        self.transform_both = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.xavier_uniform(self.transform_both[0].weight.data)
        self.transform_both[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        state = self.transform_state(state)
        action = self.transform_action(action)
        both = torch.cat([state, action], 1)
        return self.transform_both(both)


class DDPG(nn.Module):
    def __init__(self, max_buffer, state_dim, action_dim, mu, theta, sigma, actor_lr, critic_lr, batch_size, gamma, tau):
        super(DDPG, self).__init__()

        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma

        self.actor = Actor(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), actor_lr)
        self.actor_target = Actor(state_dim, action_dim)

        self.critic = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), critic_lr)
        self.critic_target = Critic(state_dim, action_dim)

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

    def sample_action(self, cx, cy, cow, cox, coy, coz, tx, ty, tow, tox, toy, toz, explore=True):
        state = Variable(torch.FloatTensor(np.transpose(np.reshape([cx,cy,cow,cox,coy,coz,tx,ty,tow,tox,toy,toz], [6*2, 1]))))
        action = self.actor(state).cpu().data.numpy()
        # add some noise to the action
        action[0][0] = utils.constrain_actions(action[0][0], 2)
        action[0][1] = utils.constrain_actions(action[0][1], 1)
        if explore:
            if random.uniform(0,1) > 0.7:
                #action = action + self.noise.sample()
                action[0][0] = random.uniform(0,2)
                action[0][1] = random.uniform(-1,1)
        return action[0][0], action[0][1]

    def learn(self):
        s, s_n, a, r = self.buffer.sample(self.batch_size)

        # The below 2 operations need to be detached since we only update eval and not targets
        a_tp1 = self.actor_target(s_n).detach()
        q_value = self.critic_target(s_n, a_tp1).squeeze().detach()
        y_target = r + self.gamma*q_value
        y_predicted = self.critic(s, a)

        critic_loss = F.smooth_l1_loss(y_predicted, y_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        a_t_pred = self.actor(s)
        q_pred = self.critic(s, a_t_pred)
        actor_loss = -1*torch.sum(q_pred)  # because we want to maximize q_pred
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_targets()
        return critic_loss, actor_loss

    def save_models(self):
        torch.save(self.actor_target.state_dict(), 'models/best_actor.model')
        torch.save(self.critic_target.state_dict(), 'models/best_critic.model')

    def load_models(self):
        self.actor_target.load_state_dict(torch.load('models/best_actor.model'))
        self.critic_target.load_state_dict(torch.load('models/best_critic.model'))
