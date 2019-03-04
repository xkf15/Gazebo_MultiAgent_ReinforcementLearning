from collections import defaultdict
from tqdm import tqdm
import numpy as np
import sys
import time
import rospy
import pickle

from gazebo_drl_env.srv import SimpleCtrl
from gazebo_drl_env.msg import control_group_msgs
from gazebo_drl_env.msg import state_group_msgs
from gazebo_drl_env.msg import control_msgs
from gazebo_drl_env.msg import state_msgs

from cv_bridge import CvBridge
import cv2

from modules import DDPG
import utils

## ------------------------------

# DDPG parameters
MAX_OPTIMIZATION_STEP = 30
TIME_DELAY = 10

STATE_DIM = 6
SENSOR_DIM = 360
ACTION_DIM = 2
MAX_EPISODES = 5000
MAX_STEPS = 20
MAX_BUFFER = 100000

USE_SHAPED_REWARD = True
USE_HER = False
USE_DIR = False
USE_TEST = True
CONTINUE_TRAIN = False
SAVE_LIDAR = False

HER_K = 8
TARGET_THRESHOLD = 0.01
TEST_ROUND = 20
TEST_EPISODE = 20
AGENT_NUMBER = 16
OBSERVATION_RANGE = 2
MIN_EXPERIMENCE_NUMBER = 3

## ------------------------------

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

## ------------------------------

# define model
model = DDPG(max_buffer=MAX_BUFFER, state_dim=STATE_DIM, sensor_dim=SENSOR_DIM, action_dim=ACTION_DIM, 
             mu=mu, theta=theta, sigma=sigma, gamma=gamma, tau=tau,
             actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size)

# define ROS service client and messages
rospy.wait_for_service('/gazebo_env_io/pytorch_io_service')
pytorch_io_service = rospy.ServiceProxy('/gazebo_env_io/pytorch_io_service', SimpleCtrl)

# load parameters and experiences
if CONTINUE_TRAIN is True:
    model.load_models()
    model.copy_weights()
    model.load_buffer()

losses = []
model.noise.reset()
succeed_time = 0
crash_time = 0
max_test_success_time = 0

# save lidar data
show_lidar_agents = defaultdict(list)
# save recent observation
last_laser_data = defaultdict(list)

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
    
    # initialize observation states
    #for i_agents in range(AGENT_NUMBER):
    #    all_current_states.group_state[i_agents] = resp_.all_group_states.group_state[i_agents]
    #    for i_ob in range(OBSERVATION_RANGE):
    #        last_laser_data[i_agents].append(resp_.all_group_states.group_state[i_agents].laserScan)

    episode_experience = defaultdict(list)
    terminate_flag = np.zeros(AGENT_NUMBER)
    experimence_mapping = np.arange(AGENT_NUMBER)
    reset_coldtime = np.zeros(AGENT_NUMBER)
    addition_experience = AGENT_NUMBER-1

    #utils.print_target_positions(resp_, AGENT_NUMBER)

    ###########################################################################
    ##### For experience collecting (Interacting with Gazebo environment) #####
    ###########################################################################
    for i_step in range(MAX_STEPS):
        # get actions for current states
        for i_agents in range(AGENT_NUMBER):
            # update last observation
            #last_laser_data[i_agents][0] = last_laser_data[i_agents][1]
            #last_laser_data[i_agents][1] = all_current_states.group_state[i_agents].laserScan

            # update current state
            all_current_states.group_state[i_agents] = resp_.all_group_states.group_state[i_agents]

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
                all_next_states.group_state[i_agents] = resp_.all_group_states.group_state[i_agents]
                
                # check terminal flags
                if all_next_states.group_state[i_agents].terminal and reset_coldtime[i_agents] == 0:
                    reset_coldtime[i_agents] = 1
                    terminate_flag[i_agents] = 1
                    all_controls.group_control[i_agents].reset = True
                    if all_next_states.group_state[i_agents].reward == 1.0:
                        succeed_time += 1
                    elif all_next_states.group_state[i_agents].reward == -1.0:
                        crash_time += 1
            time.sleep(0.1)
        resp_ = pytorch_io_service(all_controls) # make sure reset operation has been done

        # check if one agent start a new loop
        for i_flag in range(AGENT_NUMBER):
            if terminate_flag[i_flag] == 1:
                terminate_flag[i_flag] = 0 # reset flag
                addition_experience += 1
                experimence_mapping[i_flag] = addition_experience

        # temporary save experience, each loop of each agent should be saved separately
        for i_agents in range(AGENT_NUMBER):
            episode_experience[experimence_mapping[i_agents]].append(utils.combine_states(all_current_states, all_next_states, all_controls, i_agents))
            if SAVE_LIDAR is True:
                show_lidar_agents[i_agents].append(utils.remapping_laser_data(all_current_states.group_state[i_agents].laserScan))


    #####################################
    ##### For experience generating #####
    #####################################
    for i_experiment in episode_experience.keys():
        step_number = len(episode_experience[i_experiment])

        # forget too short experimence
        if step_number < MIN_EXPERIMENCE_NUMBER:
            continue

        # original HER experience
        new_goals = utils.sample_new_targets(episode_experience[i_experiment], HER_K)
        for i_step in range(step_number):
            
            # increase positive reward experience
            new_goals = utils.increase_positive_target(episode_experience[i_experiment], HER_K, i_step)

            if USE_SHAPED_REWARD:
                # save shaped reward
                new_experience = utils.shaped_reward_experience(episode_experience[i_experiment][i_step])    
            else:
                # save normal experience
                new_experience = utils.generate_experience(episode_experience[i_experiment][i_step])
            model.buffer.add(new_experience)

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


    ########################
    ##### For training #####
    ########################
    for i_learning in range(MAX_OPTIMIZATION_STEP):
        loss_c, loss_a = model.learn()
        losses.append([loss_c.cpu().data.tolist()[0], loss_a.cpu().data.tolist()[0]])
    print('Episode: %d | A-loss: %6f | C-loss: %6f | Succeed : %d | Crash : %d' % (i_episode+1, loss_c, loss_a, succeed_time, crash_time))
    
    # save model and buffer
    #model.save_models()
    model.save_buffer()

    # save lidar data
    if SAVE_LIDAR is True:
        dbfile = open('figures/lidar', 'w')
        pickle.dump(show_lidar_agents, dbfile)
        dbfile.close()


    #######################
    ##### For testing #####
    #######################
    if (i_episode+1) % TEST_EPISODE is 0 and USE_TEST is True:
        test_success_rate = 0
        print('\n==================================== Start to test ====================================')
        pbar = tqdm(total=TEST_ROUND*MAX_STEPS)
        for i_test in range(TEST_ROUND):

            # for every episode, reset environment first. Make sure getting the right position
            for i in range(AGENT_NUMBER):
                all_controls.group_control[i].linear_x = 0.0
                all_controls.group_control[i].angular_z = 0.0
                all_controls.group_control[i].reset = True
            pytorch_io_service(all_controls)

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
                            if resp_.all_group_states.group_state[i_agents].reward == 1.0:
                                test_success_rate += 1
                    time.sleep(0.1)
                resp_ = pytorch_io_service(all_controls) # make sure reset operation has been done
                pbar.update(1)

        pbar.close()
        outfile = open("test_rate.txt", 'a')
        outfile.write(str(test_success_rate))
        outfile.write('\n')
        outfile.close()
        if test_success_rate > max_test_success_time:
            max_test_success_time = test_success_rate
            model.save_model()
        print('\n==================================== Start training ====================================')

