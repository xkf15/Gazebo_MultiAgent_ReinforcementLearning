import numpy as np
import yaml
import copy

def distance(x1, y1, x2, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def get_parameters(filename):
    f = open(filename)
    y = yaml.load(f)
    return y['TERMINAL_REWARD'], y['DYNAMIC_COLLISION_REWARD'], y['SURVIVE_REWARD']

def constrain_actions(x, limit):
    x = limit if x > limit else x
    x = -limit if x < -limit else x
    return x

def print_target_positions(resp_, agent_number):
    print('-----------------------------------------')
    for i in range(agent_number):
        print('Target [%d] -- x: %f, y: %f' % (i, resp_.all_group_states.group_state[i].target_x, resp_.all_group_states.group_state[i].target_y))
    print('-----------------------------------------')

def combine_states(all_current_states, all_next_states, last_laser_data_1, last_laser_data_2, all_controls, i_agents, survive_time):
    
    survive_reward = all_next_states.group_state[i_agents].reward + survive_time[i_agents]*4.0

    return [all_current_states.group_state[i_agents].desired_x, # 0
            all_current_states.group_state[i_agents].desired_y, # 1
            all_next_states.group_state[i_agents].desired_x,    # 2
            all_next_states.group_state[i_agents].desired_y,    # 3
            all_current_states.group_state[i_agents].current_x, # 4
            all_current_states.group_state[i_agents].current_y, # 5
            all_next_states.group_state[i_agents].current_x,    # 6
            all_next_states.group_state[i_agents].current_y,    # 7
            all_next_states.group_state[i_agents].target_x,     # 8
            all_next_states.group_state[i_agents].target_y,     # 9
            all_controls.group_control[i_agents].linear_x,      # 10
            all_controls.group_control[i_agents].angular_z,     # 11
            survive_reward,                                     # 12
            remapping_laser_data(last_laser_data_1[i_agents]),                         
            remapping_laser_data(last_laser_data_2[i_agents]),
            remapping_laser_data(all_current_states.group_state[i_agents].laserScan), 
            remapping_laser_data(all_next_states.group_state[i_agents].laserScan)]

def initialze_all_states_var(temp_state, all_current_states, all_next_states, agent_number):
    for i_agents in range(agent_number):
        all_current_states.group_state.append(temp_state)
        all_next_states.group_state.append(temp_state)
    
    return all_current_states, all_next_states

def check_reset_flag(all_controls, agent_number):
    for i_agents in range(agent_number):
        if all_controls.group_control[i_agents].reset is True:
            all_controls.group_control[i_agents].reset = False

    return all_controls

def vector_normalization(v1_x, v1_y, v2_x, v2_y):
    a = v1_x-v2_x
    b = v1_y-v2_y

    return a, b
    '''
    if abs(a) < 1e-5 and abs(b) < 1e-5:
        return 0.0, 0.0
    else:
        c = (a**2+b**2)**0.5
        return a/c, b/c 
    '''

# generate buffer format in the last step
def generate_experience(old_experience):
    new_experience = []

    new_experience.append(old_experience[0])   # 0 - current desired x
    new_experience.append(old_experience[1])   # 1 - current desired y
    new_experience.append(old_experience[2])   # 2 - next desired x
    new_experience.append(old_experience[3])   # 3 - next desired y

    new_experience.append(old_experience[10])  # 4 - x speed
    new_experience.append(old_experience[11])  # 5 - z speed
    new_experience.append(old_experience[12])  # 6 - reward

    new_experience.append(old_experience[13])  # 7 - last laser data 1
    new_experience.append(old_experience[14])  # 8 - last laser data 2
    new_experience.append(old_experience[15])  # 9 - current laserscan
    new_experience.append(old_experience[16])  # 10 - next laserscan

    return new_experience

def remapping_laser_data(raw_data):

    remapping_data = np.array(raw_data)
    where_inf = np.isinf(remapping_data)
    remapping_data[where_inf] = 3.5

    # normalize laser data
    remapping_data = remapping_data - 3.5

    return remapping_data

# the uncover region of the laser scan
def laser_shape_reward_experience(episode_experience):
    new_experience = copy.deepcopy(episode_experience)

    # uncover region
    laser_uncover_region = 360*0 - np.sum(new_experience[16])
    new_experience[12] -= laser_uncover_region*0.1
    # min value
    #min_laser_value = np.min(new_experience[14])
    #new_experience[12] += (min_laser_value-3.5)*2

    return new_experience

def init_last_laser_data(agent_num):
    laser_data_1 = []
    laser_data_2 = []
    temp = np.zeros((1,360))

    for i_agent in range(agent_num):
        laser_data_1.append(temp)
        laser_data_2.append(temp)
    
    return laser_data_1, laser_data_2


'''
def update_goal_and_reward(episode_experience, new_goal, new_reward):
    new_experience = copy.deepcopy(episode_experience)

    new_experience[12] = new_reward
    # set target equal to next state
    new_experience[8] = new_goal[6]
    new_experience[9] = new_goal[7]

    return new_experience

def update_action_and_reward(episode_experience):
    new_experience = copy.deepcopy(episode_experience)

    new_experience[12] = 0.5

    # give a new commond
    new_experience[10] = -episode_experience[10]
    new_experience[11] = -episode_experience[11]

    # reverse current state and next state
    new_experience[4] = episode_experience[6]
    new_experience[5] = episode_experience[7]
    new_experience[13] = episode_experience[14]

    new_experience[6] = episode_experience[4]
    new_experience[7] = episode_experience[5]
    new_experience[14] = episode_experience[13]

    return new_experience

def shaped_reward_experience(episode_experience):
    new_experience = copy.deepcopy(episode_experience)

    #distance_collision = min(episode_experience[21])
    distance_target = distance(new_experience[8], new_experience[9], new_experience[7], new_experience[7])

    # use -log reward
    #new_experience[20] = -distance_reward+new_experience[20]
    distance_target = 0.2 if distance_target < 0.2 else distance_target
    new_experience[12] += np.log(distance_target)/np.log(0.95)

    return new_experience

def sample_new_targets(episode_experience, HER_K):
    
    if len(episode_experience) > HER_K:
        future_position = np.random.choice(np.arange(0,len(episode_experience),1), HER_K, replace=False) 
    else:
        future_position = np.arange(len(episode_experience))

    new_goals = []
    for i_k in future_position:
        new_goals.append([episode_experience[i_k][6], episode_experience[i_k][7]])

    return new_goals

def increase_positive_target(episode_experience, HER_K, i_step):
    future_position = np.random.choice(np.arange(i_step,len(episode_experience),1), HER_K, replace=True) 

    new_goals = []
    for i_k in future_position:
        new_goals.append([episode_experience[i_k][6], episode_experience[i_k][7]])
    return new_goals
'''
