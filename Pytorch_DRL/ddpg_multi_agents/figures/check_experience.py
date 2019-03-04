import numpy as np
import pickle
import matplotlib.pyplot as plt

selected_num = 7

dbfile = open('../experiences/buffer', 'r')
data = pickle.load(dbfile)

for i in range(len(data)):
    print('current position x: %.5f, y: %.5f' % (data[i][0], data[i][1]))
    print('next position    x: %.5f, y: %.5f' % (data[i][6], data[i][7]))
    #print('target position  x: %.5f, y: %.5f' % (data[i][17], data[i][18]))

    print('action           x: %.5f, z: %.5f' % (data[i][12], data[i][13]))
    print('reward:             %.5f' % data[i][14])
    print('--------------------------------------------')

