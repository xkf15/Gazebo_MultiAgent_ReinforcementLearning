import numpy as np
import yaml
import copy
import math

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

def combine_states(all_current_states, all_next_states, all_controls, i_agents, survive_time):
    
    survive_reward = all_next_states.group_state[i_agents].reward + survive_time[i_agents]*4.0
    
    current_target = target_transform(all_current_states.group_state[i_agents])
    next_target = target_transform(all_next_states.group_state[i_agents])

    desired_x = math.cos(current_target[1])
    desired_y = math.sin(current_target[1])

    return [remapping_laser_data(all_current_states.group_state[i_agents].laserScan), current_target, 
            [all_controls.group_control[i_agents].linear_x, all_controls.group_control[i_agents].angular_z],
            survive_reward,
            remapping_laser_data(all_next_states.group_state[i_agents].laserScan), next_target,
            all_next_states.group_state[i_agents].terminal,
            [desired_x, desired_y]]

'''
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
            remapping_laser_data(all_current_states.group_state[i_agents].laserScan), 
            remapping_laser_data(all_next_states.group_state[i_agents].laserScan)]
'''
    
# we use the reward reciprocal to the distance
def add_all_rewards(current_state, next_state, omega_target, reward_one_step):
    # we add a small param 0.1 to the distance, incase there will be zero on the denominator
    distance_current = 0.1 + distance(current_state.current_x, current_state.current_y, current_state.target_x, current_state.target_y)
    distance_next = 0.1 + distance(next_state.current_x, next_state.current_y, next_state.target_x, next_state.target_y)
    distance_reward = 0
    distance_reward = omega_target * (distance_current - distance_next)
    laser_reward = 5.0 * ( min(remapping_laser_data(next_state.laserScan)) - min(remapping_laser_data(current_state.laserScan)))
#    print laser_reward
#    if distance_current > 3.0:
#        distance_reward = omega_target/2 * (distance_current - distance_next)
#    else:
#        distance_reward = omega_target * (1/distance_next - 1/distance_current)
    return next_state.reward + distance_reward + laser_reward# + reward_one_step

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
'''
def generate_experience(old_experience):
    new_experience = []

    new_experience.append(old_experience[0])   # 0 - current desired x
    new_experience.append(old_experience[1])   # 1 - current desired y
    new_experience.append(old_experience[2])   # 2 - next desired x
    new_experience.append(old_experience[3])   # 3 - next desired y

    new_experience.append(old_experience[10])  # 4 - x speed
    new_experience.append(old_experience[11])  # 5 - z speed
    new_experience.append(old_experience[12])  # 6 - reward

    new_experience.append(old_experience[13])  # 7 - current laserscan
    new_experience.append(old_experience[14])  # 8 - next laserscan

    return new_experience
'''


def remapping_laser_data(raw_data):

    remapping_data = np.array(raw_data)
    where_inf = np.isinf(remapping_data)
    remapping_data[where_inf] = 3.5

    remapping_data = remapping_data - 3.5

    return remapping_data

def target_transform(state):
    target_distance = distance(state.current_x, state.current_y, state.target_x, state.target_y)
    target_yaw = math.atan2(state.target_y - state.current_y, state.target_x - state.current_x) - state.current_yaw
    # constrain the yaw to -pi ~ +pi
    if target_yaw > math.pi:
        target_yaw -= 2 * math.pi
    elif target_yaw < -math.pi:
        target_yaw += 2 * math.pi
    return [target_distance, target_yaw]

def shaped_reward_experience(episode_experience):
    new_experience = copy.deepcopy(episode_experience)

    #distance_collision = min(episode_experience[21])
    distance_target = distance(new_experience[8], new_experience[9], new_experience[7], new_experience[7])

    # use -log reward
    #new_experience[20] = -distance_reward+new_experience[20]
    distance_target = 0.2 if distance_target < 0.2 else distance_target
    new_experience[12] += np.log(distance_target)/np.log(0.95)

    return new_experience

# the uncover region of the laser scan
def laser_shape_reward_experience(episode_experience):
    new_experience = copy.deepcopy(episode_experience)

    # uncover region
    laser_uncover_region = 360*0 - np.sum(new_experience[14])
    #print(laser_uncover_region)
    new_experience[12] -= laser_uncover_region*0.1 #0.1

    # min value
    #min_laser_value = np.min(new_experience[14])
    #new_experience[12] += (min_laser_value-3.5)*2

    return new_experience

# generate laser data from the position
def generate_laser_from_pos(group_state, LASER_RANGE, ROBOT_LENGTH):
    new_group_state = copy.deepcopy(group_state)
    agent_num = len(new_group_state)
    for i in range(agent_num):
        new_group_state[i].laserScan = [float("inf") for _ in range(360)]
        for j in range(agent_num):
            if j == i:
                continue
            # calculate distance
            distance_ij = distance(new_group_state[i].current_x, new_group_state[i].current_y, new_group_state[j].current_x, new_group_state[j].current_y)
            if distance_ij >= LASER_RANGE + ROBOT_LENGTH/2 :
                continue
            angle_ij = math.atan2(new_group_state[j].current_y - new_group_state[i].current_y, new_group_state[j].current_x - new_group_state[i].current_x) - new_group_state[i].current_yaw
            # constrain the angle_ij to (0, 2 * pi)
            if angle_ij < 0:
                angle_ij += 2 * math.pi
            # calculate the robot length in laser data (occupy how much range)
            if ROBOT_LENGTH / 2 < distance_ij:
                angle_occupy_half = math.asin(ROBOT_LENGTH / 2 / distance_ij)
            else:
                angle_occupy_half = math.asin(0.5)
            upper_range = int(round((angle_ij + angle_occupy_half) * 180 / math.pi))
            lower_range = int(round((angle_ij - angle_occupy_half) * 180 / math.pi))
            # set laser
            if upper_range >= 360:
                for i_laser in range(lower_range, 360):
                    if distance_ij < new_group_state[i].laserScan[i_laser]:
                        new_group_state[i].laserScan[i_laser] = distance_ij - ROBOT_LENGTH/2
                for i_laser in range(0, upper_range - 360 + 1):
                    if distance_ij < group_state[i].laserScan[i_laser]:
                        new_group_state[i].laserScan[i_laser] = distance_ij - ROBOT_LENGTH/2
            elif lower_range < 0:
                for i_laser in range(lower_range + 360, 360):
                    if distance_ij < new_group_state[i].laserScan[i_laser]:
                        new_group_state[i].laserScan[i_laser] = distance_ij - ROBOT_LENGTH/2
                for i_laser in range(0, upper_range + 1):
                    if distance_ij < new_group_state[i].laserScan[i_laser]:
                        new_group_state[i].laserScan[i_laser] = distance_ij - ROBOT_LENGTH/2
            else:
                for i_laser in range(lower_range, upper_range + 1):
                    if distance_ij < new_group_state[i].laserScan[i_laser]:
                        new_group_state[i].laserScan[i_laser] = distance_ij - ROBOT_LENGTH/2
        new_group_state[i].laserScan = tuple(new_group_state[i].laserScan)
    return new_group_state

def comebine_sequence_data(all_next_states, i_agents):
    
    laser_data = remapping_laser_data(all_next_states.group_state[i_agents].laserScan)
    laser_uncover_region = 360.0*0.0 - np.sum(laser_data)
    reward = all_next_states.group_state[i_agents].reward - laser_uncover_region*0.1

    return laser_data, reward

def generate_hmm_sequence(episode_experience):
    X = []
    lengths = []
    rewards = []
    print('[*] Start generating hmm sequences...')
    for i_keys in episode_experience.keys():
        
        current_lens = len(episode_experience[i_keys])
        if current_lens <= 3:
            continue
        lengths.append(current_lens)
        sequence_reward = []
       
        for i_len in range(current_lens):
            
            X_sequence = np.reshape(episode_experience[i_keys][i_len][0], [1, 360])
            X = X_sequence if X == [] else np.concatenate([X, X_sequence])
            sequence_reward.append(episode_experience[i_keys][i_len][1])
        rewards.append(sequence_reward)

    return X, lengths, rewards


def from_model_to_8bits(action):
    # velocity
    vel = int(round(abs(action[0] * 7)))
    if vel is not 0:
        vel += int(action[0] < 0) << 3
    # angular
    angle = int(round(abs(action[1] * 7)))
    if angle is not 0:
        angle += int(action[1] < 0) << 3
    control_8bits = (angle << 4) + vel
    return control_8bits

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
