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
BATCH_SIZE = 128

# Hyper Parameters
LR = 0.01                 # learning rate
EPSILON = 0.3             # greedy policy
GAMMA = 0.9               # reward discount
MEMORY_CAPACITY = 50000
N_ACTIONS = 3
N_STATES = 6
MAX_EPISODE = 4000
MAX_OPTIMIZATION_STEP = 30
TARGET_REPLACE_ITER = MAX_OPTIMIZATION_STEP*3  # target update frequency
MAX_STEP_PER_EPISODE = 20
TARGET_THRESHOLD = 0.001

YAML_PARAM_PATH = '/home/dwh/3_catkin_ws/src/gym_style_gazebo/param/env.yaml'
IMAGE_COUNT = 1
HER = True
HER_K = 8

# about test
TEST_ROUND = 20
TEST_EPISODE = 10
USE_TEST = True
TIME_DELAY = 10

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def process_response(resp):
    return resp.reward, resp.terminal, \
           resp.current_x, resp.current_y, resp.orientation_w, resp.orientation_x, resp.orientation_y, resp.orientation_z, \
           resp.target_x, resp.target_y, resp.target_o_w, resp.target_o_x, resp.target_o_y, resp.target_o_z

def process_action(action):
    if action == 0:
        linear_x = 2.0; angular_z = 0.0
    elif action == 1:
        linear_x = 2.0; angular_z = 1.0
    elif action == 2:
        linear_x = 2.0; angular_z = -1.0
    else:
        print('[!] No such action...')
    return linear_x, angular_z

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1  = nn.Linear(N_STATES*2, 512)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2  = nn.Linear(512, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3  = nn.Linear(256, N_ACTIONS) 
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN(object):
    def __init__(self):
        # define two nets to implement DQN
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                 # for target updating
        self.memory_counter = 0         
        self.buffer_full = False                                    # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*3+2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, epsilon, cx, cy, cow, cox, coy, coz, tx, ty, tow, tox, toy, toz):
        x = Variable(torch.FloatTensor(np.transpose(np.reshape([cx,cy,cow,cox,coy,coz,tx,ty,tow,tox,toy,toz], [N_STATES*2, 1]))))
        randomly = False
        if np.random.uniform() > epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            randomly = True
        return action, randomly

    def store_transition(self, to_be_saved):
        transition = np.hstack(to_be_saved)

        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        if self.memory_counter >= MEMORY_CAPACITY and self.buffer_full is False:
            self.buffer_full = True

    def learn(self):
        # target parameter updating
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        selected_capacity = MEMORY_CAPACITY if self.buffer_full else self.memory_counter
        sample_index = np.random.choice(selected_capacity, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        # current state
        b_cx  = b_memory[:, 0:1]
        b_cy  = b_memory[:, 1:2]
        b_cow = b_memory[:, 2:3]
        b_cox = b_memory[:, 3:4]
        b_coy = b_memory[:, 4:5]
        b_coz = b_memory[:, 5:6]
        # next state
        b_nx  = b_memory[:, 6:7]
        b_ny  = b_memory[:, 7:8]
        b_now = b_memory[:, 8:9]
        b_nox = b_memory[:, 9:10]
        b_noy = b_memory[:, 10:11]
        b_noz = b_memory[:, 11:12]
        # target state
        b_tx  = b_memory[:, 12:13]
        b_ty  = b_memory[:, 13:14]
        b_tow = b_memory[:, 14:15]
        b_tox = b_memory[:, 15:16]
        b_toy = b_memory[:, 16:17]
        b_toz = b_memory[:, 17:18]
        # reward and action
        b_a = Variable(torch.LongTensor(b_memory[:, 18:19].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, 19:20]))

        # reshape 
        b_s  = [b_cx, b_cy, b_cow, b_cox, b_coy, b_coz, b_tx, b_ty, b_tow, b_tox, b_toy, b_toz]
        b_s_ = [b_nx, b_ny, b_now, b_nox, b_noy, b_noz, b_tx, b_ty, b_tow, b_tox, b_toy, b_toz]
        b_s  = Variable(torch.FloatTensor(np.transpose(np.reshape(b_s,  [N_STATES*2, BATCH_SIZE]))))
        b_s_ = Variable(torch.FloatTensor(np.transpose(np.reshape(b_s_, [N_STATES*2, BATCH_SIZE]))))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
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
succeed_time = 0

# main loop
print('Start training...')
for i_episode in range(MAX_EPISODE):
    # for every episode, reset environment first. Make sure getting the right position
    pytorch_io_service(0.0, 0.0, True)
    time.sleep(0.5)
    response = pytorch_io_service(0.0, 0.0, False)
    _, _, cx, cy, cow, cox, coy, coz, tx, ty, tow, tox, toy, toz = process_response(response)
    
    step_count = 0
    episode_experience = []
    print('Target x: %f, Target y: %f' % (tx, ty))
    # for experience generation, interacting with environment
    for i_step in range(MAX_STEP_PER_EPISODE):
        a, randomly = dqn.choose_action(EPSILON, cx, cy, cow, cox, coy, coz, tx, ty, tow, tox, toy, toz)

        # implement action and get response
        linear_x, angular_z = process_action(a)
        for t in range(TIME_DELAY):
            response_ = pytorch_io_service(linear_x, angular_z, False)
            r, terminal, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz = process_response(response_)
            if terminal == 1: break
            time.sleep(0.1)

        if r == 1: succeed_time += 1

        # modified reward
        #r = -distance(nx, ny, tx, ty)*10
        
        episode_experience.append((cx, cy, cow, cox, coy, coz, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz, a, r))
        step_count += 1
        cx, cy, cow, cox, coy, coz = nx, ny, now, nox, noy, noz
        if randomly:
            print('[', a, ']', end='')
        else:
            print(a, end='')
        if terminal:
            break
    print('')

    # for experience collection, review all steps
    for i_step in range(step_count):
        cx, cy, cow, cox, coy, coz, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz, a, r = episode_experience[i_step]
        to_be_saved = (cx, cy, cow, cox, coy, coz, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz, a, r)
        dqn.store_transition(to_be_saved)
        if HER:
            for i_goal in range(HER_K):
                # generate one number form i_steo to step_count
                future = np.random.randint(i_step, step_count) 
                _, _, _, _, _, _, tx_new, ty_new, _, _, _, _, _, _, _, _, _, _, _, _ = episode_experience[future]
                r = 1.0 if distance(tx_new, ty_new, nx, ny) <= TARGET_THRESHOLD else 0.0
                to_be_saved = (cx, cy, cow, cox, coy, coz, nx, ny, now, nox, noy, noz, tx_new, ty_new, tow, tox, toy, toz, a, r)
                dqn.store_transition(to_be_saved)

    # for optimization
    mean_loss = 0
    for i_learning in range(MAX_OPTIMIZATION_STEP):
        mean_loss += dqn.learn().data.numpy()[0]
    print('Episode: %d | Loss: %6f | Step: %d | Succeed: %d | Epsilon: %6f' % (i_episode, mean_loss/MAX_OPTIMIZATION_STEP, step_count, succeed_time, EPSILON))

    # for test
    if (i_episode+1) % TEST_EPISODE is 0 and USE_TEST is True:
        tmp_rate = 0
        for i_test in range(TEST_ROUND):
            pytorch_io_service(0.0, 0.0, True)
            time.sleep(0.5)
            response = pytorch_io_service(0.0, 0.0, False)
            _, _, cx, cy, cow, cox, coy, coz, tx, ty, tow, tox, toy, toz = process_response(response)
 
            for i_step in range(MAX_STEP_PER_EPISODE):
                a, _ = dqn.choose_action(0.0, cx, cy, cow, cox, coy, coz, tx, ty, tow, tox, toy, toz)

                linear_x, angular_z = process_action(a)
                for t in range(TIME_DELAY):
                    response_ = pytorch_io_service(linear_x, angular_z, False)
                    r, terminal, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz = process_response(response_)
                    if terminal == 1: break
                    time.sleep(0.1)

                cx, cy, cow, cox, coy, coz = nx, ny, now, nox, noy, noz
                if r == 1:
                    tmp_rate += 1
                if terminal:
                    break;
            time.sleep(0.5)
            print('finish one test, this is {}/{}'.format(i_test+1, TEST_ROUND), end='\r')
            sys.stdout.flush()
        print('')
        outfile = open("test_rate.txt", 'a')
        outfile.write(str(tmp_rate))
        outfile.write('\n')
        outfile.close()


