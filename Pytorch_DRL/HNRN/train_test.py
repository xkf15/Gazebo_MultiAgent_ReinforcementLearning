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
import cv2

from HNRN import HNRN
import utils

## ---------------------------------

# 1 - train target driven actor
# 2 - train collision avoidance actor
# 3 - differentail driver
TRAIN_TYPE = 1

# Params for environments
AGENT_NUMBER = 4
STATE_DIM = 2
SENSOR_DIM = 2
ACTION_DIM = 2
TIME_DELAY = 2

# Params for training
MAX_EPISODES = 1

# use absolute coordinate to generate
GENERATE_LASER_FORM_POS = True
ROBOT_LENGTH = 0.25
LASER_RANGE = 3.5


omega_target = 2.0

## ---------------------------------

# define model
# model = HNRN()

# define ROS service client and messages
rospy.wait_for_service('/gazebo_env_io/pytorch_io_service')
pytorch_io_service = rospy.ServiceProxy('/gazebo_env_io/pytorch_io_service', SimpleCtrl)
TERMINAL_REWARD, COLLISION_REWARD, SURVIVE_REWARD = utils.get_parameters('../../gazebo_drl_env/param/env.yaml')

print('[*] State	Dimensions : {}'.format(STATE_DIM))
print('[*] Sensor Dimensions : {}'.format(SENSOR_DIM))
print('[*] Action Dimensions : {}'.format(ACTION_DIM))
print('[*] Agent	Number : {}'.format(AGENT_NUMBER))
print('\n============================== Start training ==============================')


for i_episode in range(MAX_EPISODES):

    

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
        current_control.linear_x = 0.1
        current_control.angular_z = 0.0
        current_control.reset = True
        all_controls.group_control.append(current_control)
    pytorch_io_service(all_controls)

    # after reseting, publish a new action for all agents
    for i in range(AGENT_NUMBER):
        all_controls.group_control[i].reset = False
    resp_ = pytorch_io_service(all_controls)
    if GENERATE_LASER_FORM_POS is True:
        resp_.all_group_states.group_state = utils.generate_laser_from_pos(resp_.all_group_states.group_state, LASER_RANGE, ROBOT_LENGTH)
    time.sleep(0.001)

    episode_experience = defaultdict(list)
    terminate_flag = np.zeros(AGENT_NUMBER)
    survive_time = np.zeros(AGENT_NUMBER)
    experimence_mapping = np.arange(AGENT_NUMBER)
    reset_coldtime = np.zeros(AGENT_NUMBER)
    addition_experience = AGENT_NUMBER

    for i in range(15):
        resp_ = pytorch_io_service(all_controls)
        if GENERATE_LASER_FORM_POS is True:
            resp_.all_group_states.group_state = utils.generate_laser_from_pos(resp_.all_group_states.group_state, LASER_RANGE, ROBOT_LENGTH)
        for i_agents in range(AGENT_NUMBER):
            # update current state
            all_current_states.group_state[i_agents] = copy.deepcopy(resp_.all_group_states.group_state[i_agents])
            all_controls.group_control[i_agents].reset = False
            reset_coldtime[i_agents] = 0

    # implement action and get response during some time
    # when one agent is terminated, start a new step after reseting
        for t in range(TIME_DELAY):
            resp_ = pytorch_io_service(all_controls)
            if GENERATE_LASER_FORM_POS is True:
                resp_.all_group_states.group_state = utils.generate_laser_from_pos(resp_.all_group_states.group_state, LASER_RANGE, ROBOT_LENGTH)
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

            time.sleep(0.001)
        resp_ = pytorch_io_service(all_controls) # make sure reset operation has been done
        if GENERATE_LASER_FORM_POS is True:
            resp_.all_group_states.group_state = utils.generate_laser_from_pos(resp_.all_group_states.group_state, LASER_RANGE, ROBOT_LENGTH)

        if all_next_states.group_state[0].terminal == False:
            all_next_states.group_state[0].reward = utils.add_all_rewards(all_current_states.group_state[0], all_next_states.group_state[0], omega_target)
        print(all_next_states.group_state[0].target_x, all_next_states.group_state[0].target_y)
        print(all_next_states.group_state[0].reward)
        time.sleep(1)


