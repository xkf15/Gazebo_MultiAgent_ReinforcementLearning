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
BATCH_SIZE = 16

# Hyper Parameters
LR = 0.001                 # learning rate
EPSILON = 0.9              # greedy policy
GAMMA = 0.9                # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 5000
N_ACTIONS = 5
N_STATES = IMAGE_WIDTH*IMAGE_HEIGHT
MAX_EPISODE = 4000
MAX_OPTIMIZATION_STEP = 40
TARGET_THRESHOLD = 0.001

YAML_PARAM_PATH = '/home/dwh/3_catkin_ws/src/gym_style_gazebo/param/env.yaml'
IMAGE_COUNT = 1
HER = True
HER_K = 8

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def get_target_position(filename):
    f = open(filename)
    y = yaml.load(f)
    return y['TARGET_X'], y['TARGET_Y'], y['TERMINAL_REWARD']

def process_response(resp):
    global IMAGE_COUNT
    cv_image = CvBridge().imgmsg_to_cv2(resp.depth_img, "32FC1")
    cv_image_resized = cv2.resize(cv_image,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_CUBIC)
    cv_image_resized = np.nan_to_num(cv_image_resized)
    cv_image_resized = 255*cv_image_resized/(np.max(cv_image_resized)-np.min(cv_image_resized))
    cv2.imwrite('./images/' + str(IMAGE_COUNT) + '.jpg', cv_image_resized)
    IMAGE_COUNT += 1
    return cv_image_resized, resp.reward, resp.terminal, resp.current_x, resp.current_y

def process_action(action):
    if action == 0:
        linear_x = 1.5; angular_z = 0.0
    elif action == 1:
        linear_x = 0.7; angular_z = 0.7
    elif action == 2:
        linear_x = 0.7; angular_z = 1.5
    elif action == 3:
        linear_x = 0.7; angular_z = -0.7
    elif action == 4:
        linear_x = 0.7; angular_z = -1.5
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
        self.fc1 = nn.Linear(2640, 500)
        self.relu3 = nn.ReLU()
        self.fc2  = nn.Linear(500, 100)
        self.relu4 = nn.ReLU()
        self.fc3  = nn.Linear(104, N_ACTIONS)
        self.soft1 = nn.Softmax()

        self.apply(weights_init)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)

    def forward(self, x, c, t):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(-1, 2640)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = torch.cat((x, c, t), -1)
        x = self.soft1(self.fc3(x))
        return x

class DQN(object):
    def __init__(self):
        # define two nets to implement DQN
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+8))         # initialize memory
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

    def store_transition(self, s, x, y, a, r, s_, x_, y_, t_x, t_y):
        s = list(np.reshape(s, [1, IMAGE_HEIGHT*IMAGE_WIDTH]))
        s_ = list(np.reshape(s_, [1, IMAGE_HEIGHT*IMAGE_WIDTH]))
        transition = np.hstack((s[0], x, y, x_, y_, t_x, t_y, a, r, s_[0]))

        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter updating
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        # current depth image
        b_s = b_memory[:, :N_STATES]

        # current position
        b_x = b_memory[:, N_STATES:N_STATES+1]
        b_y = b_memory[:, N_STATES+1:N_STATES+2] 
        # next position
        b_xn = b_memory[:, N_STATES+2:N_STATES+3]
        b_yn = b_memory[:, N_STATES+3:N_STATES+4]
        # target position
        b_tx = b_memory[:, N_STATES+4:N_STATES+5]
        b_ty = b_memory[:, N_STATES+5:N_STATES+6]
        # reward and action
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES+6:N_STATES+7].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+7:N_STATES+8]))
        # next depth image
        b_s_ = b_memory[:, -N_STATES:]

        # reshape to image size
        b_p = Variable(torch.FloatTensor(np.reshape([b_x, b_y], [BATCH_SIZE, 2])))
        b_p_ = Variable(torch.FloatTensor(np.reshape([b_xn, b_yn], [BATCH_SIZE, 2])))
        b_t = Variable(torch.FloatTensor(np.reshape([b_tx, b_ty], [BATCH_SIZE, 2])))
        b_s = Variable(torch.FloatTensor(np.reshape(b_s, [BATCH_SIZE, 1, IMAGE_WIDTH, IMAGE_HEIGHT])))
        b_s_ = Variable(torch.FloatTensor(np.reshape(b_s_, [BATCH_SIZE, 1, IMAGE_WIDTH, IMAGE_HEIGHT])))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s, b_p, b_t).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_, b_p_, b_t).detach()    # detach from graph, don't backpropagate
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
target_x, target_y, target_reward = get_target_position(YAML_PARAM_PATH)

# main loop
print('Collecting experience...')
for i_episode in range(MAX_EPISODE):
    # for every episode, reset environment first
    response = pytorch_io_service(0.0, 0.0, True)
    state, _, _, c_x, c_y = process_response(response)
    succeed_time = 0
    step_count = 0
    episode_experience = []
    terminal = False

    # for experience generation, interacting with environment
    while terminal is False:
        time.sleep(0.5)
        action = dqn.choose_action(state, c_x, c_y, target_x, target_y)
        print(action, end='')

        # implement action and get response
        linear_x, angular_z = process_action(action)
        response_ = pytorch_io_service(linear_x, angular_z, False)
        state_, reward, terminal, c_x_, c_y_ = process_response(response_)
        episode_experience.append((state, c_x, c_y, action, reward, state_, c_x_, c_y_, target_x, target_y))
        
        if reward == target_reward:
            succeed_time += 1
        step_count += 1
        state, c_x, c_y = state_, c_x_, c_y_
        #print('Generating experience: ', step_count, end='\r')
        #sys.stdout.flush()
    print('')

    # for experience collection, review all steps
    for i_step in range(step_count):
        s, x, y, a, r, s_n, x_n, y_n, t_x, t_y = episode_experience[i_step]
        dqn.store_transition(s, x, y, a, r, s_n, x_n, y_n, t_x, t_y)
        if HER:
            for i_goal in range(HER_K):
                future = np.random.randint(i_step, step_count) 
                _, _, _, _, _, _, t_x_new, t_y_new, _, _ = episode_experience[future]
                r = 1 if distance(t_x_new, t_y_new, x_n, y_n) < TARGET_THRESHOLD else 0
                dqn.store_transition(s, x, y, a, r, s_n, x_n, y_n, t_x_new, t_y_new)

    # for optimization
    mean_loss = 0
    for i_learning in range(MAX_OPTIMIZATION_STEP):
        mean_loss += dqn.learn().data.numpy()
    print('Episode: {} | Mean loss is: {} | Max step is: {} | Succeed time: {}'.format(i_episode, mean_loss/MAX_OPTIMIZATION_STEP, step_count, succeed_time))
