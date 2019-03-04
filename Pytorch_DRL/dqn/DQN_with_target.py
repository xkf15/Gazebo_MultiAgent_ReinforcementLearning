from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import time

# for ROS
import rospy
from gym_style_gazebo.srv import SimpleCtrl
from cv_bridge import CvBridge
import cv2
import yaml

'''
====  Actions mapping table  ====
1 -- linear_x=0.5, angular_z=0.0
2 -- linear_x=0.1, angular_z=0.5
3 -- linear_x=0.1, angular_z=1.0
4 -- linear_x=0.1, angular_z=-0.5
5 -- linear_x=0.1, angular_z=-1.0
'''

# size of data, big size will increase the consume of memory
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48
BATCH_SIZE = 64

# Hyper Parameters
LR = 0.0001                  # learning rate
EPSILON = 0.8             # greedy policy
GAMMA = 0.98                # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 5000
N_ACTIONS = 5
N_STATES = IMAGE_WIDTH*IMAGE_HEIGHT
MAX_EPISODE = 4000
MAX_STEP_PER_EPISODE = 200

YAML_PARAM_PATH = '/home/dwh/3_catkin_ws/src/gym_style_gazebo/param/env.yaml'
IMAGE_COUNT = 1

def get_target_position(filename):
    f = open(filename)
    y = yaml.load(f)
    return y['TARGET_X'], y['TARGET_Y']

def process_response(resp):
    global IMAGE_COUNT
    cv_image = CvBridge().imgmsg_to_cv2(resp.depth_img, "32FC1")
    cv_image_resized = cv2.resize(cv_image,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_CUBIC)
    cv_image_resized = np.nan_to_num(cv_image_resized)
    cv_image_resized = 255*cv_image_resized/(np.max(cv_image_resized)-np.min(cv_image_resized))
    #cv2.imwrite('./images/' + str(IMAGE_COUNT) + '.jpg', cv_image_resized)
    #IMAGE_COUNT += 1
    return cv_image_resized, resp.reward, resp.terminal, resp.current_x, resp.current_y


def process_action(action):
    if action == 0:
        linear_x = 0.7; angular_z = 0.0
    elif action == 1:
        linear_x = 0.5; angular_z = 0.5
    elif action == 2:
        linear_x = 0.5; angular_z = 1.0
    elif action == 3:
        linear_x = 0.5; angular_z = -0.5
    elif action == 4:
        linear_x = 0.5; angular_z = -1.0
    else:
        print('[!] No such action...')
    return linear_x, angular_z

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2) 
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(2644, 500)
        self.soft1 = nn.Softmax()
        self.fc2  = nn.Linear(500, N_ACTIONS)

        self.apply(weights_init)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)

    def forward(self, x, c, t):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(-1, 2640)
        x = torch.cat((x, c, t), -1)
        x = self.soft1(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN(object):
    def __init__(self):
        # define two nets to implement DQN
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+6))         # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, cx, cy, tx, ty):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        x = x.view(-1, 1, IMAGE_WIDTH, IMAGE_HEIGHT) 
        c = Variable(torch.FloatTensor(np.reshape([cx,cy], [1,2])))
        t = Variable(torch.FloatTensor(np.reshape([tx,ty], [1,2])))

        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x, c, t)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] 
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_, c_x, c_y, t_x, t_y):
        s = list(np.reshape(s, [1, IMAGE_HEIGHT*IMAGE_WIDTH]))
        s_ = list(np.reshape(s_, [1, IMAGE_HEIGHT*IMAGE_WIDTH]))

        transition = np.hstack((s[0], a, r, c_x, c_y, t_x, t_y, s_[0]))

        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        if self.memory_counter < MEMORY_CAPACITY:
            print('[*] Saving experience, and this is {}/{} memory experience...'.format(self.memory_counter+1, MEMORY_CAPACITY), end='\r')
            sys.stdout.flush()

    def learn(self):
        # target parameter updating
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = b_memory[:, :N_STATES]
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_cx = b_memory[:, N_STATES+2:N_STATES+3]
        b_cy = b_memory[:, N_STATES+3:N_STATES+4]
        b_c = Variable(torch.FloatTensor(np.reshape([b_cx, b_cy], [BATCH_SIZE, 2])))
        b_tx = b_memory[:, N_STATES+4:N_STATES+5]
        b_ty = b_memory[:, N_STATES+5:N_STATES+6]
        b_t = Variable(torch.FloatTensor(np.reshape([b_tx, b_ty], [BATCH_SIZE, 2])))
        b_s_ = b_memory[:, -N_STATES:]

        # reshape to image size
        b_s = Variable(torch.FloatTensor(np.reshape(b_s, [BATCH_SIZE, 1, IMAGE_WIDTH, IMAGE_HEIGHT])))
        b_s_ = Variable(torch.FloatTensor(np.reshape(b_s_, [BATCH_SIZE, 1, IMAGE_WIDTH, IMAGE_HEIGHT])))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s, b_c, b_t).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_, b_c, b_t).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

''' ----------------------------------------------------------------------------------------------------------- '''

# define DQN network
dqn = DQN()
# define ROS service client
rospy.wait_for_service('/gazebo_env_io/pytorch_io_service')
pytorch_io_service = rospy.ServiceProxy('/gazebo_env_io/pytorch_io_service', SimpleCtrl)
target_x, target_y = get_target_position(YAML_PARAM_PATH)

# main loop
print('Collecting experience...')
for i_episode in range(MAX_EPISODE):
    # for every episode, reset environment first
    response = pytorch_io_service(0.0, 0.0, True)
    state, _, _, current_x, current_y = process_response(response)
    ep_r = 0

    for i_step in range(MAX_STEP_PER_EPISODE):
        time.sleep(0.1)
        action = dqn.choose_action(state, current_x, current_y, target_x, target_y)

        # implement action and get response
        linear_x, angular_z = process_action(action)
        response_ = pytorch_io_service(linear_x, angular_z, False)
        state_, reward, terminal, current_x, current_y = process_response(response_)

        # save one transition
        dqn.store_transition(state, action, reward, state_, current_x, current_y, target_x, target_y)

        ep_r += reward
        if dqn.memory_counter > MEMORY_CAPACITY:
            loss = dqn.learn()
            print('Action is :{} | Loss is :{}'.format(action, loss.data.numpy()))
            if terminal: 
                print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))

        if terminal:
            break
        state = state_