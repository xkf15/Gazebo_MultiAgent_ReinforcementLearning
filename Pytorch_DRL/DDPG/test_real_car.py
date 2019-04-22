import time
import pickle
import random

from DDPG import DDPG
import utils

## ------------------------------

# 1 - train target driven actor
# 2 - train collision avoidance actor
# 3 - differential driver
TRAIN_TYPE = 2

TIME_DELAY = 2#5
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
sigma = 0.8#1.0#0.2

# learning rate
actor_lr = 1e-5 # 1e-4
critic_lr = 1e-4 # 1e-3
batch_size = 256

# update parameters
gamma = 0.99
tau = 0.001
hmm_state = 10

## ------------------------------

# define model
model = DDPG(max_buffer=0, state_dim=STATE_DIM, sensor_dim=SENSOR_DIM, target_dim=TARGET_DIM, action_dim=ACTION_DIM, 
             mu=mu, theta=theta, sigma=sigma, gamma=gamma, tau=tau, train_type=TRAIN_TYPE,
             actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, hmm_state=hmm_state)

# load parameters of pre-trained model
model.load_models('models/best_ca_actor.model', 'models/best_ca_critic.model')
model.copy_weights()

# random target within the boundary of environment
temp_target = [4.0, 4.0]

class state_struct():
    def __init__(self, laserScan, current_x, current_y, current_yaw, target_x, target_y):
        self.laserScan = laserScan
        self.current_x = current_x
        self.current_y = current_y
        self.current_yaw = current_yaw
        self.target_x = target_x
        self.target_y = target_y

temp_state = state_struct([float("inf") for _ in range(360)], 0.0, 0.0, 0.5, temp_target[0], temp_target[1])

def inference(state):

    action = [0,0]

    # Do Inference
    action[0], action[1] = model.sample_action(current_state=state, explore=False)

    return action


