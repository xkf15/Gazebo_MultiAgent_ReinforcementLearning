import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

f = open('./loss.txt','r')
f2 = open('../3.20/loss.txt','r')
data = pickle.load(f)
data2 = pickle.load(f2)
f.close()
f2.close()

data_len = len(data)
c_loss = []
c_smooth = []
gamma = 0.99

data_len2 = len(data2)
c_loss2 = []
c_smooth2 = []

a_loss = []
crash_time = []
crash_time_average = []
 
for i in range(28000):
    c_loss.append(data[i][0])
    if i is 0:
        c_smooth.append(data[i][0])
    c_smooth.append(data[i][0] * (1-gamma) + gamma * c_smooth[-1])
    a_loss.append(data[i][1])
    if i % 30 == 0:
		    crash_time.append(data[i][2])
		    if i >= 750:
		        crash_time_average.append(sum(crash_time[-5:-1])/5)


for i in range(28000):
    c_loss2.append(data2[i][0])
    if i is 0:
        c_smooth2.append(data2[i][0])
    c_smooth2.append(data2[i][0] * (1-gamma) + gamma * c_smooth2[-1])
    a_loss.append(data2[i][1])


plt.title('Critic Loss', fontsize='24')
plt.plot(c_smooth2, color = 'red', label = 'No Pretrain')
plt.plot(c_smooth, color = 'green', label = 'Pretrain')
plt.legend(fontsize='22')
plt.ylabel('Critic Loss', fontsize='22')
plt.xlabel('Training Times', fontsize='22')
#plt.title('actor_loss')
#plt.plot(a_loss)
#plt.title('crash times')
#plt.plot(crash_time_average)
plt.show()
