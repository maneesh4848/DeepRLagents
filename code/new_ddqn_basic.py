#!/usr/bin/env python
from __future__ import print_function
import pdb
import random
from random import choice
import numpy as np
import time
from datetime import datetime
from game_util import preprocessImg, play_epoch, play_episode
from game_util import img_rows, img_cols, img_channels

import os

import json
from keras import backend as K

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import tensorflow as tf
import time
from networks.keras_networks import Networks
from networks.ddqn import DoubleDQNAgent
from logger import FileWriter


def main():

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config("../scenarios/simpler_basic.cfg")
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    save_stats_file_name = "../statistics/new_ddqn_basic_ph1_" + timestamp_str + ".txt"
    stats_file_logger = FileWriter(save_stats_file_name, False, True)
    save_model_file_name = "../models/new_ddqn_basic_ph1.h5" #Saved every epoch

    game_state = game.get_state()

    action_size = game.get_available_buttons_size()

    state_size = (img_rows, img_cols, img_channels)
    params = {'gamma': 0.99, 'lr': 0.0001, 'eps': 1.0, 'init_eps': 1.0,
              'final_eps': 0.0001,'batch_size': 32,'obs': 500,'exp': 5000,
              'fpa': 4,'target_freq': 300,'timestep_train': 10,'max_memory': 500}
    agent = DoubleDQNAgent(state_size, action_size, params)

    agent.set_model(Networks.new_dqn(state_size, action_size, params["lr"]))
    agent.set_target_model(Networks.new_dqn(state_size, action_size, params["lr"]))
    
    x_t = game_state.screen_buffer # 480 x 640
    x_t = preprocessImg(x_t, size=(img_rows, img_cols))
    s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    # Start training
    t = 0

    epochs = 5
    games_per_epoch = 5
    temp_buffer=[]
    for epoch in range(epochs):
        print("Epoch:", epoch)
        t = play_epoch(game, agent, epoch, games_per_epoch, t, stats_file_logger)

        # save progress every epoch
        print("Now we save model")
        agent.save_model(save_model_file_name)

        # Reset rolling stats buffer
        score_buffer, ammo_buffer, health_buffer = [], [], [] 

        # Write Rolling Statistics to file
        stats_file_logger.writeline("mavg_score: " + str(agent.mavg_score))
        stats_file_logger.writeline("var_score: " + str(agent.var_score))
        stats_file_logger.writeline("mavg_ammo_left: " + str(agent.mavg_ammo_left))
        stats_file_logger.writeline("mavg_kill_counts: " + str(agent.mavg_kill_counts))
        stats_file_logger.writeline("")

    print("Training done.... test time")
    game = DoomGame()
    game.load_config("../scenarios/simpler_basic.cfg")
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()
    action_size = game.get_available_buttons_size()
    episodes_to_watch=10
    for i in range(episodes_to_watch):
        play_episode(game, agent, action_size, 0, i, t, False)

        score = game.get_total_reward()    
        print("Total score: ",score)

if __name__ == "__main__":
    main()

