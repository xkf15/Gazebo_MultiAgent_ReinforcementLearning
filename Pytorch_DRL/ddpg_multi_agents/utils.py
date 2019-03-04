import numpy as np
import yaml
import copy

def distance(x1, y1, x2, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def get_parameters(filename):
    f = open(filename)
    y = yaml.load(f)
    return y['TARGET_X'], y['TARGET_Y'], y['TARGET_ORIENTATION_W'], y['TARGET_ORIENTATION_X'], y['TARGET_ORIENTATION_Y'], y['TARGET_ORIENTATION_Z']

def constrain_actions(x, limit):
    x = limit if x > limit else x
    x = -limit if x < -limit else x
    return x

def print_target_positions(resp_, agent_number):
    print('-----------------------------------------')
    for i in range(agent_number):
        print('Target [%d] -- x: %f, y: %f' % (i, resp_.all_group_states.group_state[i].target_x, resp_.all_group_states.group_state[i].target_y))
    print('-----------------------------------------')

def combine_states(all_current_states, all_next_states, all_controls, i_agents):

    return [all_current_states.group_state[i_agents].current_x,
            all_current_states.group_state[i_agents].current_y,
            all_current_states.group_state[i_agents].orientation_w,
            all_current_states.group_state[i_agents].orientation_x,
            all_current_states.group_state[i_agents].orientation_y,
            all_current_states.group_state[i_agents].orientation_z,
            all_next_states.group_state[i_agents].current_x,
            all_next_states.group_state[i_agents].current_y,
            all_next_states.group_state[i_agents].orientation_w,
            all_next_states.group_state[i_agents].orientation_x,
            all_next_states.group_state[i_agents].orientation_y,
            all_next_states.group_state[i_agents].orientation_z,
            all_current_states.group_state[i_agents].target_x,
            all_current_states.group_state[i_agents].target_y,
            all_current_states.group_state[i_agents].target_o_w,
            all_current_states.group_state[i_agents].target_o_x,
            all_current_states.group_state[i_agents].target_o_y,
            all_current_states.group_state[i_agents].target_o_z,
            all_controls.group_control[i_agents].linear_x,
            all_controls.group_control[i_agents].angular_z,
            all_next_states.group_state[i_agents].reward,
            remapping_laser_data(all_current_states.group_state[i_agents].laserScan), 
            remapping_laser_data(all_next_states.group_state[i_agents].laserScan)]

def update_goal_and_reward(episode_experience, new_goal, new_reward):
    new_experience = copy.deepcopy(episode_experience)

    new_experience[20] = new_reward
    # set target equal to next state
    new_experience[12] = new_goal[0]
    new_experience[13] = new_goal[1]

    return new_experience

def update_action_and_reward(episode_experience):
    new_experience = copy.deepcopy(episode_experience)

    new_experience[20] = 0.5

    # give a new commond
    new_experience[18] = -episode_experience[18]
    new_experience[19] = -episode_experience[19]

    # reverse current state and next state
    new_experience[0] = episode_experience[6]
    new_experience[1] = episode_experience[7]
    new_experience[21] = episode_experience[22]

    new_experience[6] = episode_experience[0]
    new_experience[7] = episode_experience[1]
    new_experience[22] = episode_experience[21]

    return new_experience


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

    cvx, cvy = vector_normalization(old_experience[12], old_experience[13], old_experience[0], old_experience[1])
    nvx, nvy = vector_normalization(old_experience[12], old_experience[13], old_experience[6], old_experience[7])


    new_experience.append(cvx)
    new_experience.append(cvy)

    new_experience.append(old_experience[2])
    new_experience.append(old_experience[3])
    new_experience.append(old_experience[4])
    new_experience.append(old_experience[5])

    new_experience.append(nvx)
    new_experience.append(nvy)

    new_experience.append(old_experience[8])
    new_experience.append(old_experience[9])
    new_experience.append(old_experience[10])
    new_experience.append(old_experience[11])

    new_experience.append(old_experience[18])  # 12 - x speed
    new_experience.append(old_experience[19])  # 13 - z speed
    new_experience.append(old_experience[20])  # 14 - reward

    new_experience.append(old_experience[21])  # 15
    new_experience.append(old_experience[22])  # 16

    #new_experience.append(old_experience[12])  # 17 - target x (optional)
    #new_experience.append(old_experience[13])  # 18 - target y (optional)

    return new_experience

def remapping_laser_data(raw_data):

    remapping_data = np.array(raw_data)
    where_inf = np.isinf(remapping_data)
    remapping_data[where_inf] = 3.5

    return remapping_data

def shaped_reward_experience(episode_experience):
    new_experience = copy.deepcopy(episode_experience)

    #distance_collision = min(episode_experience[21])
    distance_reward = distance(new_experience[12], new_experience[13], new_experience[0], new_experience[1])
    new_experience[20] = -distance_reward+new_experience[20]

    new_experience = generate_experience(new_experience)
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