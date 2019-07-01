import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

f = open('./loss.txt','r')
data = pickle.load(f)
f.close()

data_len = len(data)
c_loss = []
a_loss = []
crash_time = []
crash_time_average = []

for i in range(data_len):
    c_loss.append(data[i][0])
    a_loss.append(data[i][1])
    if i % 30 == 0:
        crash_time.append(data[i][2])
        if i >= 750:
            crash_time_average.append(sum(crash_time[-5:-1])/5)

#plt.title('critic_loss')
#plt.plot(c_loss)
plt.title('8 Agents Training')
plt.plot(a_loss)
plt.ylabel('-Q value')
plt.xlabel('Training Times')
#plt.title('crash times')
#plt.plot(crash_time_average)
plt.show()
