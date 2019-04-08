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
success_time = []

for i in range(data_len):
    c_loss.append(data[i][0])
    a_loss.append(data[i][1])
    if i % 10 == 0:
        crash_time.append(data[i][3])
        success_time.append(data[i][2])

#plt.title('critic_loss')
#plt.plot(c_loss)
#plt.title('actor_loss')
#plt.plot(a_loss)
plt.title('success times')
plt.plot(success_time)
plt.show()
