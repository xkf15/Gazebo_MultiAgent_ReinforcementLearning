## Gazebo_MultiAgent_ReinforcementLearning
The environment is still chaging, this project is not finished.

# Steps to begin:
1. Set parameters in the gazebo_drl_env/param/env.yaml. Remember to change AGENT_NUMBER and MAP_TYPE.
2. Change parameters in the Pytorch_DRL/DDPG/train_DDPG.py. Remember to change AGENT_NUMBER.
3. Run train_DDPG.py using python 2.7

# Debug Notes:
When compile the package for the first time, an error says it can not find PythonRL.h. Then, recompile after all other packages are finished compiling. Sometimes it may have errors again, then recompile again after all compilings.
