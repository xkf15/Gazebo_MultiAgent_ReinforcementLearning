import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import time
import math

def distance(x1, y1, x2, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

MAX_EPISODES = 74
success = 0
crash = 0

trajectory_distance = []
velocity_x = []
velocity_z = []

dbfile_trajectory = open('./trajectory', 'r')
dbfile_keypoint = open('./keypoint', 'r')


for i_episodes in range(MAX_EPISODES):

    
    trajectory = pickle.load(dbfile_trajectory)  
    keypoint = pickle.load(dbfile_keypoint)

    agent_num = len(trajectory)    
    for i_agent in range(agent_num):
        distance_trajectory = 0
        len_trajectory = len(trajectory[i_agent])
        len_keypoint = len(keypoint[i_agent])
        # calculate distance
        for i_trajectory in range(1, len_trajectory):
            distance_trajectory += distance(trajectory[i_agent][i_trajectory-1][0], trajectory[i_agent][i_trajectory-1][1], trajectory[i_agent][i_trajectory][0], trajectory[i_agent][i_trajectory][1])
        trajectory_distance.append(distance_trajectory)
        # save velocity
        for i_keypoint in range(len_keypoint):
            velocity_x.append(abs(keypoint[i_agent][i_keypoint][2]))
            velocity_z.append(abs(keypoint[i_agent][i_keypoint][3]))

dbfile_trajectory.close()
dbfile_keypoint.close()

print('\nTrajectory Distance: Mean {} | Var {}'.format(np.mean(trajectory_distance), math.sqrt(np.var(trajectory_distance))))
print('Velocity X: Mean {} | Var {}'.format(np.mean(velocity_x), math.sqrt(np.var(velocity_x))))
print('Velocity Z: Mean {} | Var {}'.format(np.mean(velocity_z), math.sqrt(np.var(velocity_z))))


    
