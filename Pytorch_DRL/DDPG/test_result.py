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
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2

from DDPG import DDPG
import utils

## ------------------------------

# 1 - train target driven actor
# 2 - train collision avoidance actor
# 3 - differential driver
TRAIN_TYPE = 2

TIME_DELAY = 1#5
STATE_DIM = 2
SENSOR_DIM = 360
ACTION_DIM = 2
TARGET_THRESHOLD = 0.01
AGENT_NUMBER = 4

MAX_EPISODES = 100
MAX_STEPS = 999999#30

# use absolute coordinate to generate
GENERATE_LASER_FORM_POS = True
ROBOT_LENGTH = 0.25
LASER_RANGE = 3.5

# new reward params
omega_target = 10.0
TARGET_DIM = 2
reward_one_step = -0.1 # get penalty each step

## ------------------------------

# noise parameters
mu = 0 
theta = 2
sigma = 1.0#0.2

# learning rate
actor_lr = 1e-5 # 1e-4
critic_lr = 1e-4 # 1e-3
batch_size = 256

# update parameters
gamma = 0.99
tau = 0.001

## ------------------------------

# define model
model = DDPG(max_buffer=0, state_dim=STATE_DIM, sensor_dim=SENSOR_DIM, target_dim=TARGET_DIM, action_dim=ACTION_DIM, 
             mu=mu, theta=theta, sigma=sigma, gamma=gamma, tau=tau, train_type=TRAIN_TYPE,
             actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size)

# define ROS service client and messages
rospy.wait_for_service('/gazebo_env_io/pytorch_io_service')
pytorch_io_service = rospy.ServiceProxy('/gazebo_env_io/pytorch_io_service', SimpleCtrl)
TERMINAL_REWARD, COLLISION_REWARD, SURVIVE_REWARD = utils.get_parameters('../../gazebo_drl_env/param/env.yaml')

# load parameters of pre-trained model
model.load_models('models/best_ca_actor.model', 'models/best_ca_critic.model')
model.copy_weights()
model.noise.reset()

print('[*] State  Dimensions : {}'.format(STATE_DIM))
print('[*] Sensor Dimensions : {}'.format(SENSOR_DIM))
print('[*] Action Dimensions : {}'.format(ACTION_DIM))
print('[*] Agent  Number : {}'.format(AGENT_NUMBER))
print('\n==================================== Start Navigation ====================================')

# save trajectory
agent_trajectory = defaultdict(list)
agent_keypoint = defaultdict(list)
first_time = 1

success_time = 0
collision_time = 0

for i_episode in range(MAX_EPISODES):

    # save trajectory
    agent_trajectory = defaultdict(list)
    agent_keypoint = defaultdict(list)

    # record start time
    time_start = time.time()

    crash_time_episode = 0
    succeed_time_episode = 0
    success_agent_num = 0
    collision_happen = False
    terminal_flag = [False for i in range(AGENT_NUMBER)]

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
    if GENERATE_LASER_FORM_POS is True:
        resp_.all_group_states.group_state = utils.generate_laser_from_pos(resp_.all_group_states.group_state, LASER_RANGE, ROBOT_LENGTH)
    time.sleep(0.0015)

    terminate_flag = np.zeros(AGENT_NUMBER)
    survive_time = np.zeros(AGENT_NUMBER)
    reset_coldtime = np.zeros(AGENT_NUMBER)

    # interacting with the environment
    for i_step in range(MAX_STEPS):
        # get actions for current states
        for i_agents in range(AGENT_NUMBER):
            # update current state
            all_current_states.group_state[i_agents] = copy.deepcopy(resp_.all_group_states.group_state[i_agents])
            all_controls.group_control[i_agents].linear_x, all_controls.group_control[i_agents].angular_z = \
            model.sample_action(current_state=all_current_states.group_state[i_agents], explore=False)
            all_controls.group_control[i_agents].reset = False
            reset_coldtime[i_agents] = 0
            #if first_time == 1:
            # save [x, y, vx, vy]
            agent_keypoint[i_agents].append([all_current_states.group_state[i_agents].current_x, all_current_states.group_state[i_agents].current_y, all_controls.group_control[i_agents].linear_x, all_controls.group_control[i_agents].angular_z])


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
                    # all_controls.group_control[i_agents].reset = True
                    if all_next_states.group_state[i_agents].reward == TERMINAL_REWARD:
                        if terminal_flag[i_agents] == False:
                            succeed_time_episode += 1
                        terminal_flag[i_agents] = True
                        all_controls.group_control[i_agents].reset = True
                        for tt in range(AGENT_NUMBER):
                            if terminal_flag[tt] == False:
                                all_controls.group_control[i_agents].reset = False
                    if all_next_states.group_state[i_agents].reward == COLLISION_REWARD:
                        all_controls.group_control[i_agents].reset = True
                        collision_happen = True#collision_time += 1
                
#                if first_time == 1:
                agent_trajectory[i_agents].append([all_next_states.group_state[i_agents].current_x, all_next_states.group_state[i_agents].current_y])

            time.sleep(0.0015)
        resp_ = pytorch_io_service(all_controls) # make sure reset operation has been done
        if GENERATE_LASER_FORM_POS is True:
            resp_.all_group_states.group_state = utils.generate_laser_from_pos(resp_.all_group_states.group_state, LASER_RANGE, ROBOT_LENGTH)

        # check if one agent start a new loop
        for i_agents in range(AGENT_NUMBER):
            if terminate_flag[i_agents] == 1:
                terminate_flag[i_agents] = 0 # reset flag

        # reset terminal flags
        reset_terminal_flags = False
        for tt in range(AGENT_NUMBER):
            if terminal_flag[tt] == True:
                if tt == AGENT_NUMBER-1:
                    success_time += 1
                    reset_terminal_flags = True
            else:
               break
        if collision_happen == True:
            reset_terminal_flags = True
            collision_time += 1
        if reset_terminal_flags == True:
            terminal_flag = [False for i in range(AGENT_NUMBER)]
            break
    
    # calculate time
    time_end = time.time()
    time_episode = time_end - time_start

    if first_time == 1:
        dbfile = open('figures_utils/trajectory', 'w')
        pickle.dump(agent_trajectory, dbfile)
        dbfile.close()    
        dbfile = open('figures_utils/keypoint', 'w')
        pickle.dump(agent_keypoint, dbfile)
        dbfile.close()   
        first_time = 0
    else:
        dbfile = open('figures_utils/trajectory', 'a')
        pickle.dump(agent_trajectory, dbfile)
        dbfile.close()
        dbfile = open('figures_utils/keypoint', 'a')
        pickle.dump(agent_keypoint, dbfile)
        dbfile.close()   
        #pbar.update()
    print('[*] Successfully arriving target times: {} | Crash times: {} | Steps this episode: {} | Time this episode: {}'.format(success_time, collision_time, i_step, time_episode))
    outfile = open("navigation_result.txt", 'a')
    outfile.write(str(success_time)+' '+str(collision_time)+' '+str(i_step)+' '+str(time_episode)+'\n')
    outfile.close()



