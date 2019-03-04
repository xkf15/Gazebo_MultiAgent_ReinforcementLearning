from gym_style_gazebo.srv import SimpleCtrl
from cv_bridge import CvBridge
from modules import DDPG
from tqdm import tqdm

import numpy as np
import sys
import time
import rospy
import cv2
import utils

# rl parameters
MAX_OPTIMIZATION_STEP = 30
TIME_DELAY = 10

STATE_DIM = 12
ACTION_DIM = 2
MAX_EPISODES = 5000
MAX_STEPS = 20
MAX_BUFFER = 100000

USE_HER = True
HER_K = 8
TARGET_THRESHOLD = 0.001
TEST_ROUND = 20
TEST_EPISODE = 20
USE_TEST = True
YAML_PARAM_PATH = '/home/dwh/1_normal_workspace/catkin_ws/src/gym_style_gazebo/param/env.yaml'

# noise parameters
mu = 0 
theta = 2
sigma = 0.2
# learning rate
actor_lr = 1e-4
critic_lr = 1e-3
batch_size = 128
# update parameters
gamma = 0.99
tau = 0.001

# define model
model = DDPG(MAX_BUFFER, STATE_DIM, ACTION_DIM, mu, theta, sigma, actor_lr, critic_lr, batch_size, gamma, tau)
# define ROS service client
rospy.wait_for_service('/gazebo_env_io/pytorch_io_service')
pytorch_io_service = rospy.ServiceProxy('/gazebo_env_io/pytorch_io_service', SimpleCtrl)

# load parameters
#model.load_models()
#model.copy_weights()

print('[*] State Dimensions : ', STATE_DIM)
print('[*] Action Dimensions : ', ACTION_DIM)
print('[*] Start training...')
losses = []
model.noise.reset()
succeed_time = 0

for i_episode in range(MAX_EPISODES):
    # for every episode, reset environment first. Make sure getting the right position
    pytorch_io_service(0.0, 0.0, True)
    time.sleep(0.5)
    resp_ = pytorch_io_service(0.0, 0.0, False)
    _, _, cx, cy, cow, cox, coy, coz, tx, ty, tow, tox, toy, toz = utils.process_response(resp_)

    step_count = 0
    episode_experience = []
    print('Target x: %f, Target y: %f' % (tx, ty))

    # for experience generating
    for i_step in range(MAX_STEPS):
        a_x, a_z = model.sample_action(cx, cy, cow, cox, coy, coz, tx, ty, tow, tox, toy, toz)
        # implement action and get response
        for t in range(TIME_DELAY):
            resp_ = pytorch_io_service(a_x, a_z, False)
            r, terminal, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz = utils.process_response(resp_)
            if terminal == 1:
                break
            time.sleep(0.1)
        if r == 1: 
            succeed_time += 1
        
        # modified reward
        #r = -utils.distance(nx, ny, tx, ty)*10

        # temporary save experience
        episode_experience.append((cx, cy, cow, cox, coy, coz, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz, a_x, a_z, r))
        step_count += 1
        cx, cy, cow, cox, coy, coz = nx, ny, now, nox, noy, noz
        if terminal: 
            break

    # for experience collecting
    for i_step in range(step_count):
        cx, cy, cow, cox, coy, coz, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz, a_x, a_z, r = episode_experience[i_step]
        model.buffer.add((cx, cy, cow, cox, coy, coz, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz, a_x, a_z, r))
        if USE_HER:
            for i_goal in range(HER_K):
                # generate one number form i_steo to step_count
                future = np.random.randint(i_step, step_count) 
                _, _, _, _, _, _, tx_new, ty_new, _, _, _, _, _, _, _, _, _, _, _, _, _ = episode_experience[future]
                r_new = 1.0 if utils.distance(tx_new, ty_new, nx, ny) <= TARGET_THRESHOLD else 0.0
                model.buffer.add((cx, cy, cow, cox, coy, coz, nx, ny, now, nox, noy, noz, tx_new, ty_new, tow, tox, toy, toz, a_x, a_z, r_new))

    # for training 
    for i_learning in range(MAX_OPTIMIZATION_STEP):
        _loss_c, _loss_a = model.learn()
        losses.append([_loss_c.cpu().data.tolist()[0], _loss_a.cpu().data.tolist()[0]])
    print('Episode: %d | A-loss: %6f | C-loss: %6f | Step: %d | Succeed: %d' % (i_episode, _loss_c, _loss_a, step_count, succeed_time))
    model.save_models()

    # for test
    if (i_episode+1) % TEST_EPISODE is 0 and USE_TEST is True:
        tmp_rate = 0
        print('Start to test...')
        pbar = tqdm(total=TEST_ROUND)
        for i_test in range(TEST_ROUND):
            # for every episode, reset environment first. Make sure getting the right position
            pytorch_io_service(0.0, 0.0, True)
            time.sleep(0.5)
            resp_ = pytorch_io_service(0.0, 0.0, False)
            _, _, cx, cy, cow, cox, coy, coz, tx, ty, tow, tox, toy, toz = utils.process_response(resp_)
            episode_experience = []

            # for experience generating
            for i_step in range(MAX_STEPS):
                # close exploring when testing
                a_x, a_z = model.sample_action(cx, cy, cow, cox, coy, coz, tx, ty, tow, tox, toy, toz, explore=False)
                # implement action and get response
                for t in range(TIME_DELAY):
                    resp_ = pytorch_io_service(a_x, a_z, False)
                    r, terminal, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz = utils.process_response(resp_)
                    if terminal == 1: 
                        break
                    time.sleep(0.1)
                if r == 1: 
                    tmp_rate += 1
                
                # temporary save experience
                episode_experience.append((cx, cy, cow, cox, coy, coz, nx, ny, now, nox, noy, noz, tx, ty, tow, tox, toy, toz, a_x, a_z, r))
                cx, cy, cow, cox, coy, coz = nx, ny, now, nox, noy, noz
                if terminal: break
            pbar.update(1)

        pbar.close()
        outfile = open("test_rate.txt", 'a')
        outfile.write(str(tmp_rate))
        outfile.write('\n')
        outfile.close()
