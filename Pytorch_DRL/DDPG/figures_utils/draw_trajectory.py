import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

dbfile = open('./trajectory', 'r')
data = pickle.load(dbfile)
dbfile.close()

dbfile = open('./keypoint', 'r')
keypoint = pickle.load(dbfile)
dbfile.close()

#plt.figure(figsize=(10,6))
for i_agents in range(len(data)):
    position_x = []
    position_y = []
    for i_step in range(len(data[i_agents])):
        position_x.append(data[i_agents][i_step][0])  
        position_y.append(data[i_agents][i_step][1])      

    plt.plot(position_x, position_y, linewidth=5, alpha=0.4)        

#plt.figure(figsize=(10,6))
for i_agents in range(len(keypoint)):

    if i_agents == 0:
        current_c = 'b'
    elif i_agents == 1:
        current_c = 'y'
    elif i_agents == 2:
        current_c = 'g'
    elif i_agents == 3:
        current_c = 'r'

    position_x = []
    position_y = []
    for i_step in range(len(keypoint[i_agents])):
        if i_step != 0:
            plt.scatter(keypoint[i_agents][i_step][0], keypoint[i_agents][i_step][1], linewidth=float(i_step)/3.0, c=current_c, alpha=float(i_step)/float(len(keypoint[i_agents])))     

plt.axis('off')
plt.show()
