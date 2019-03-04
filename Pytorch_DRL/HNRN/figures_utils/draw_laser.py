import numpy as np
import matplotlib.pyplot as plt
import pickle

dbfile = open('../experiences/buffer', 'r')
data = pickle.load(dbfile)

plt.figure(figsize=(10,6))

theta = np.arange(0, 2*np.pi, 2*np.pi/360)
for i in range(len(data)):
    ax1 = plt.subplot(121, projection='polar')
    plt.title('Current laser scan', verticalalignment='bottom')
    ax1.set_theta_zero_location('S')
    ax2 = plt.subplot(122, projection='polar')
    plt.title('Next laser scan', verticalalignment='bottom')
    ax2.set_theta_zero_location('S')
    ax1.plot(theta, data[i][15], lw=2)
    ax2.plot(theta, data[i][16], lw=2)
    strr = 'Action x: ' + str(data[i][12])
    strr2 = 'Action z: ' + str(data[i][13])
    plt.text(0, 4.5, strr, horizontalalignment='center')
    plt.text(0, 5, strr2, horizontalalignment='center')
    strr = 'Reward: ' + str(data[i][14])
    plt.text(0, 5.5, strr, horizontalalignment='center')

    plt.pause(1)
    plt.clf()

plt.show()
