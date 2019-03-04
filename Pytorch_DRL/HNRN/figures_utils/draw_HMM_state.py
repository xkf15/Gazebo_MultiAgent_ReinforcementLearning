import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


dbfile = open('../models/HMM/training_data', 'r')
training_data = pickle.load(dbfile)
dbfile.close()
dbfile = open('../models/HMM/hidden_states', 'r')
hidden_states = pickle.load(dbfile)

#plt.figure(figsize=(10,6))

theta = np.arange(0, 2*np.pi, 2*np.pi/360)
max_show = np.zeros(10)
MAX_SHOW = 20

for i in range(len(training_data)):
    
    ax1 = plt.subplot(251, projection='polar')
    #plt.title('State 1', verticalalignment='bottom')
    ax1.set_theta_zero_location('S')
    plt.thetagrids([30, 60, 120, 150, 210, 240, 300, 330])

    ax2 = plt.subplot(252, projection='polar')
    #plt.title('State 2', verticalalignment='bottom')
    ax2.set_theta_zero_location('S')
    plt.thetagrids([30, 60, 120, 150, 210, 240, 300, 330])
    
    ax3 = plt.subplot(253, projection='polar')
    #plt.title('State 3', verticalalignment='bottom')
    ax3.set_theta_zero_location('S')
    plt.thetagrids([30, 60, 120, 150, 210, 240, 300, 330])

    ax4 = plt.subplot(254, projection='polar')
    #plt.title('State 4', verticalalignment='bottom')
    ax4.set_theta_zero_location('S')
    plt.thetagrids([30, 60, 120, 150, 210, 240, 300, 330])

    ax5 = plt.subplot(255, projection='polar')
    #plt.title('State 5', verticalalignment='bottom')
    ax5.set_theta_zero_location('S')
    plt.thetagrids([30, 60, 120, 150, 210, 240, 300, 330])

    ax6 = plt.subplot(256, projection='polar')
    #plt.title('State 6', verticalalignment='bottom')
    ax6.set_theta_zero_location('S')
    plt.thetagrids([30, 60, 120, 150, 210, 240, 300, 330])

    ax7 = plt.subplot(257, projection='polar')
    #plt.title('State 7', verticalalignment='bottom')
    ax7.set_theta_zero_location('S')
    plt.thetagrids([30, 60, 120, 150, 210, 240, 300, 330])

    ax8 = plt.subplot(258, projection='polar')
    #plt.title('State 8', verticalalignment='bottom')
    ax8.set_theta_zero_location('S')
    plt.thetagrids([30, 60, 120, 150, 210, 240, 300, 330])

    ax9 = plt.subplot(259, projection='polar')
    #plt.title('State 9', verticalalignment='bottom')
    ax9.set_theta_zero_location('S')
    plt.thetagrids([30, 60, 120, 150, 210, 240, 300, 330])

    ax10 = plt.subplot(2,5,10, projection='polar')
    #plt.title('State 10', verticalalignment='bottom')
    ax10.set_theta_zero_location('S')
    plt.thetagrids([30, 60, 120, 150, 210, 240, 300, 330])


    if hidden_states[i] == 0 and max_show[0] <= MAX_SHOW:
        ax1.plot(theta, training_data[i], lw=2)
        ax1.set_thetagrids(np.arange(0.0, 360.0, 30.0))
        ax1.set_rlim(-3.5, 0.0)
        max_show[0] += 1
    elif hidden_states[i] == 1 and max_show[1] <= MAX_SHOW:
        ax2.plot(theta, training_data[i], lw=2)
        ax2.set_thetagrids(np.arange(0.0, 360.0, 30.0))
        ax2.set_rlim(-3.5, 0.0)
        max_show[1] += 1
    elif hidden_states[i] == 2 and max_show[2] <= MAX_SHOW:
        ax3.plot(theta, training_data[i], lw=2)
        ax3.set_thetagrids(np.arange(0.0, 360.0, 30.0))
        ax3.set_rlim(-3.5, 0.0)
        max_show[2] += 1
    elif hidden_states[i] == 3 and max_show[3] <= MAX_SHOW:
        ax4.plot(theta, training_data[i], lw=2)
        ax4.set_thetagrids(np.arange(0.0, 360.0, 30.0))
        ax4.set_rlim(-3.5, 0.0)
        max_show[3] += 1
    elif hidden_states[i] == 4 and max_show[4] <= MAX_SHOW:
        ax5.plot(theta, training_data[i], lw=2)
        ax5.set_thetagrids(np.arange(0.0, 360.0, 30.0))
        ax5.set_rlim(-3.5, 0.0)
        max_show[4] += 1
    elif hidden_states[i] == 5 and max_show[5] <= MAX_SHOW:
        ax6.plot(theta, training_data[i], lw=2)
        ax6.set_thetagrids(np.arange(0.0, 360.0, 30.0))
        ax6.set_rlim(-3.5, 0.0)
        max_show[5] += 1
    elif hidden_states[i] == 6 and max_show[6] <= MAX_SHOW:
        ax7.plot(theta, training_data[i], lw=2)
        ax7.set_thetagrids(np.arange(0.0, 360.0, 30.0))
        ax7.set_rlim(-3.5, 0.0)
        max_show[6] += 1
    elif hidden_states[i] == 7 and max_show[7] <= MAX_SHOW:
        ax8.plot(theta, training_data[i], lw=2)
        ax8.set_thetagrids(np.arange(0.0, 360.0, 30.0))
        ax8.set_rlim(-3.5, 0.0)
        max_show[7] += 1
    elif hidden_states[i] == 8 and max_show[8] <= MAX_SHOW:
        ax9.plot(theta, training_data[i], lw=2)
        ax9.set_thetagrids(np.arange(0.0, 360.0, 30.0))
        ax9.set_rlim(-3.5, 0.0)
        max_show[8] += 1
    elif hidden_states[i] == 9 and max_show[9] <= MAX_SHOW:
        ax10.plot(theta, training_data[i], lw=2)
        ax10.set_thetagrids(np.arange(0.0, 360.0, 30.0))
        ax10.set_rlim(-3.5, 0.0)
        max_show[9] += 1

    plt.pause(0.1)
    #plt.clf()

plt.show()
