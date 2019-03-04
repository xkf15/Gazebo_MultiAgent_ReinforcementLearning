import numpy as np
import yaml

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def get_target_position(filename):
    f = open(filename)
    y = yaml.load(f)
    return y['TARGET_X'], y['TARGET_Y'], y['TARGET_ORIENTATION_W'], y['TARGET_ORIENTATION_X'], y['TARGET_ORIENTATION_Y'], y['TARGET_ORIENTATION_Z']

def process_response(resp):
    return resp.reward, resp.terminal, \
           resp.current_x, resp.current_y, resp.orientation_w, resp.orientation_x, resp.orientation_y, resp.orientation_z, \
           resp.target_x, resp.target_y, resp.target_o_w, resp.target_o_x, resp.target_o_y, resp.target_o_z

def constrain_actions(x, limit):
    x = limit if x > limit else x
    x = -limit if x < -limit else x
    return x
    