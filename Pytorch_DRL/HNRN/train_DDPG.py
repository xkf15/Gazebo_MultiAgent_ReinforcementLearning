from collections import defaultdict
from tqdm import tqdm
import numpy as np
import sys
import time
import rospy
import pickle
import copy

from gazebo_drl_env.srv import SimpleCtrl
from gazebo_drl_env.msg import control_group_msgs
from gazebo_drl_env.msg import state_group_msgs
from gazebo_drl_env.msg import control_msgs
from gazebo_drl_env.msg import state_msgs

from cv_bridge import CvBridge

# remove the ros packages path, or cv2 can not work
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2

from HNRN import HNRN
import utils

## ------------------------------

# 1 - train target driven actor
# 2 - train collision avoidance actor
# 3 - differential driver
TRAIN_TYPE = 2

MAX_OPTIMIZATION_STEP = 30
TIME_DELAY = 5
STATE_DIM = 2
SENSOR_DIM = 360
ACTION_DIM = 2
MAX_EPISODES = 5000
MAX_STEPS = 30
MAX_BUFFER = 50000
HER_K = 8
TARGET_THRESHOLD = 0.01
TEST_ROUND = 20
TEST_EPISODE = 20 if TRAIN_TYPE == 1 else 40
AGENT_NUMBER = 16
OBSERVATION_RANGE = 2
MIN_EXPERIMENCE_NUMBER = 3

USE_HER = False
USE_DIR = False
USE_TEST = True
USE_SHAPED_REWARD = False
USE_LASER_REWARD = True
USE_SURVIVE_REWARD = False

CONTINUE_TRAIN = False

## ------------------------------

# noise parameters
mu = 0 
theta = 2
sigma = 0.2
# learning rate
actor_lr = 1e-4
critic_lr = 1e-3
batch_size = 256
# update parameters
gamma = 0.99
tau = 0.001
hmm_state = 20

## ------------------------------

# define model
model = HNRN(max_buffer=MAX_BUFFER, state_dim=STATE_DIM, sensor_dim=SENSOR_DIM, action_dim=ACTION_DIM, 
             mu=mu, theta=theta, sigma=sigma, gamma=gamma, tau=tau, train_type=TRAIN_TYPE,
             actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, hmm_state=hmm_state)

# define ROS service client and messages
rospy.wait_for_service('/gazebo_env_io/pytorch_io_service')
pytorch_io_service = rospy.ServiceProxy('/gazebo_env_io/pytorch_io_service', SimpleCtrl)
TERMINAL_REWARD, COLLISION_REWARD, SURVIVE_REWARD = utils.get_parameters('../../gazebo_drl_env/param/env.yaml')

# load parameters and experiences
if CONTINUE_TRAIN is True:
    model.load_models()
    model.copy_weights()
    #model.load_buffer()

losses = []
model.noise.reset()
succeed_time = 0
crash_time = 0
max_test_success_time = 0
max_total_reward = -99999


print('[*] State  Dimensions : {}'.format(STATE_DIM))
print('[*] Sensor Dimensions : {}'.format(SENSOR_DIM))
print('[*] Action Dimensions : {}'.format(ACTION_DIM))
print('[*] Agent  Number : {}'.format(AGENT_NUMBER))
print('\n==================================== Start training ====================================')

for i_episode in range(MAX_EPISODES):

    # placeholder for states
    all_controls = control_group_msgs()
    all_current_states = state_group_msgs()
    all_next_states = state_group_msgs()
    temp_state = control_msgs()
    # inistalize current and next states
    all_current_states, all_next_states = utils.initialze_all_states_var(temp_state, all_current_states, all_next_states, AGENT_NUMBER)

    # for every episode, reset environment first. Make sure getting the right position.
    # All control messages are setted to 0.0 at the first time, and retains the value unless get new control commond for that agent.
    for i in range(AGENT_NUMBER):
        # control message should be created every time 
        current_control = control_msgs()
        current_control.linear_x = 0.0
        current_control.angular_z = 0.0
        current_control.reset = True
        all_controls.group_control.append(current_control)
    pytorch_io_service(all_controls)
    
    # after reseting, publish a new action for all agents
    for i in range(AGENT_NUMBER):
        all_controls.group_control[i].reset = False
    resp_ = pytorch_io_service(all_controls)
    time.sleep(0.2)
    

    episode_experience = defaultdict(list)
    terminate_flag = np.zeros(AGENT_NUMBER)
    survive_time = np.zeros(AGENT_NUMBER)
    experimence_mapping = np.arange(AGENT_NUMBER)
    reset_coldtime = np.zeros(AGENT_NUMBER)
    addition_experience = AGENT_NUMBER


    ###########################################################################
    ##### For experience collecting (Interacting with Gazebo environment) #####
    ###########################################################################
    for i_step in range(MAX_STEPS):
        # get actions for current states
        for i_agents in range(AGENT_NUMBER):
            # update current state
            all_current_states.group_state[i_agents] = copy.deepcopy(resp_.all_group_states.group_state[i_agents])
            all_controls.group_control[i_agents].linear_x, all_controls.group_control[i_agents].angular_z = \
            model.sample_action(current_state=all_current_states.group_state[i_agents], explore=True)
            all_controls.group_control[i_agents].reset = False
            reset_coldtime[i_agents] = 0

        # implement action and get response during some time
        # when one agent is terminated, start a new step after reseting
        for t in range(TIME_DELAY):
            resp_ = pytorch_io_service(all_controls)
            all_controls = utils.check_reset_flag(all_controls, AGENT_NUMBER)
            for i_agents in range(AGENT_NUMBER):
                # update all states
                if (terminate_flag[i_agents] == 0):
                    all_next_states.group_state[i_agents] = copy.deepcopy(resp_.all_group_states.group_state[i_agents])

                # check terminal flags
                if all_next_states.group_state[i_agents].terminal and reset_coldtime[i_agents] == 0:
                    reset_coldtime[i_agents] = 1
                    terminate_flag[i_agents] = 1
                    all_controls.group_control[i_agents].reset = True
                    if all_next_states.group_state[i_agents].reward == TERMINAL_REWARD:
                        succeed_time += 1
                    elif all_next_states.group_state[i_agents].reward == COLLISION_REWARD:
                        crash_time += 1
            time.sleep(0.1)
        resp_ = pytorch_io_service(all_controls) # make sure reset operation has been done

        # check if one agent start a new loop
        for i_agents in range(AGENT_NUMBER):
            if terminate_flag[i_agents] == 1:
                terminate_flag[i_agents] = 0 # reset flag
                experimence_mapping[i_agents] = addition_experience
                addition_experience += 1
                survive_time[i_agents] = 0
            else:
                if USE_SURVIVE_REWARD is True:
                    survive_time[i_agents] += 1
            # temporary save experience, each loop of each agent should be saved separately
            episode_experience[experimence_mapping[i_agents]].append(utils.combine_states(all_current_states, all_next_states, all_controls, i_agents, survive_time))


    #####################################
    ##### For experience generating #####
    #####################################
    for i_experiment in episode_experience.keys():
        step_number = len(episode_experience[i_experiment])

        # forget too short experimence
        if step_number < MIN_EXPERIMENCE_NUMBER:
            continue

        # original HER experience
        #new_goals = utils.sample_new_targets(episode_experience[i_experiment], HER_K)
        for i_step in range(step_number):

            # increase positive reward experience
            #new_goals = utils.increase_positive_target(episode_experience[i_experiment], HER_K, i_step)
            new_experience = episode_experience[i_experiment][i_step]
            if TRAIN_TYPE == 1 and USE_SHAPED_REWARD is True:
                new_experience = utils.shaped_reward_experience(new_experience)    
            if TRAIN_TYPE == 2 and USE_LASER_REWARD is True:
                new_experience = utils.laser_shape_reward_experience(new_experience)   
            
            new_experience = utils.generate_experience(new_experience)
            model.buffer.add(new_experience)

            '''
            # save directional experience
            if USE_DIR and episode_experience[i_experiment][i_step][20] == -1.0:
                # collision happens
                new_experience = utils.update_action_and_reward(episode_experience[i_experiment][i_step])
                new_experience = utils.generate_experience(new_experience)
                model.buffer.add(new_experience)

            # save HER experimence
            if USE_HER:
                for i_goal in new_goals:
                    # use distance to tolearant the error
                    if utils.distance(i_goal[0], i_goal[1], episode_experience[i_experiment][i_step][6], episode_experience[i_experiment][i_step][7]) <= TARGET_THRESHOLD:
                        new_reward = 1.0 
                    else:
                        new_reward = 0.0 

                    # save eaperimence
                    new_experience = utils.update_goal_and_reward(episode_experience[i_experiment][i_step], i_goal, new_reward)
                    new_experience = utils.generate_experience(new_experience)
                    model.buffer.add(new_experience)
            '''


    ########################
    ##### For training #####
    ########################
    if TRAIN_TYPE is not 3:
        for i_learning in range(MAX_OPTIMIZATION_STEP):
            loss_c, loss_a = model.learn()
            losses.append([loss_c.cpu().data.tolist(), loss_a.cpu().data.tolist()])
        print('Episode: %d | A-loss: %6f | C-loss: %6f | Succeed : %d | Crash : %d' % (i_episode+1, loss_a, loss_c, succeed_time, crash_time))
        dbfile = open('result/loss/loss.txt', 'w')
        pickle.dump(losses, dbfile)
        dbfile.close()
        #model.save_buffer()


    #######################
    ##### For testing #####
    #######################
    if (i_episode+1) % TEST_EPISODE is 0 and USE_TEST is True:
        test_success_rate = 0
        total_reward = 0
        print('\n==================================== Start to test ====================================')
        pbar = tqdm(total=TEST_ROUND*MAX_STEPS)
        for i_test in range(TEST_ROUND):

            # for every episode, reset environment first. Make sure getting the right position
            for i in range(AGENT_NUMBER):
                all_controls.group_control[i].linear_x = 0.0
                all_controls.group_control[i].angular_z = 0.0
                all_controls.group_control[i].reset = True
            resp_ = pytorch_io_service(all_controls)

            # after reseting, publish a new action for all agents
            for i in range(AGENT_NUMBER):
                all_controls.group_control[i].reset = False
            time.sleep(0.2)
            resp_ = pytorch_io_service(all_controls)
            reset_coldtime = np.zeros(AGENT_NUMBER)

            for i_step in range(MAX_STEPS):
                # get actions for current states
                for i_agents in range(AGENT_NUMBER):
                    all_controls.group_control[i_agents].linear_x, all_controls.group_control[i_agents].angular_z = \
                    model.sample_action(current_state=resp_.all_group_states.group_state[i_agents], explore=False)
                    all_controls.group_control[i_agents].reset = False
                    reset_coldtime[i_agents] = 0

                # implement action and get response during some time
                # when one agent is terminated, start a new step after reseting
                for t in range(TIME_DELAY):
                    resp_ = pytorch_io_service(all_controls)
                    all_controls = utils.check_reset_flag(all_controls, AGENT_NUMBER)
                    for i_agents in range(AGENT_NUMBER):
                        # check terminal flags
                        if resp_.all_group_states.group_state[i_agents].terminal and reset_coldtime[i_agents] == 0:
                            reset_coldtime[i_agents] = 1
                            all_controls.group_control[i_agents].reset = True
                            total_reward += resp_.all_group_states.group_state[i_agents].reward

                            if resp_.all_group_states.group_state[i_agents].reward == TERMINAL_REWARD:
                                test_success_rate += 1
                    time.sleep(0.1)
                resp_ = pytorch_io_service(all_controls) # make sure reset operation has been done
                pbar.update(1)

        pbar.close()
        outfile = open("test_rate.txt", 'a')
        if TRAIN_TYPE is not 2:
            outfile.write(str(test_success_rate))
            if test_success_rate >= max_test_success_time:
                max_test_success_time = test_success_rate
                model.save_models()
        else:
            outfile.write(str(total_reward/TEST_ROUND))
            if total_reward >= max_total_reward:
                max_total_reward = total_reward
                model.save_models()
        outfile.write('\n')
        outfile.close()

        print('\n==================================== Start training ====================================')

