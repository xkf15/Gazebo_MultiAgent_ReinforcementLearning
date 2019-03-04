import numpy as np
import pickle
import matplotlib.pyplot as plt

selected_num = 7

dbfile = open('lidar', 'r')
data = pickle.load(dbfile)

new_data = np.reshape(data[selected_num], [len(data[selected_num]), 360])

print(new_data)

plt.imshow(new_data[0:2000,:]*10)
plt.show()
dbfile.close()
