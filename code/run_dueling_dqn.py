#!/usr/bin/env python
from __future__ import print_function
import pdb
import numpy as np
from datetime import datetime
from game_util import preprocessImg, get_argparser, play_epoch, play_episode
from game_util import img_rows, img_cols, img_channels

from keras import backend as K

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import tensorflow as tf
import time
from networks.keras_networks import Networks
from networks.ddqn import DoubleDQNAgent
from logger import *


def main():

    parser = get_argparser()
    args = parser.parse_args()

    log_writer = None
    if args.log != "":
        log_writer = LogWriter(args.log)
    else:
        log_writer = LogWriter()

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config(args.scenario)
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    save_stats_file_name = "../statistics/new_ddqn_basic_ph1_"
    if args.per:
        save_stats_file_name += "per_"
    save_stats_file_name += timestamp_str + ".txt"
    stats_file_logger = FileWriter(save_stats_file_name, False, True)
    save_model_file_name = args.model #Saved every epoch

    game_state = game.get_state()
    action_size = game.get_available_buttons_size()
    state_size = (img_rows, img_cols, img_channels)
    params = {'gamma': 0.99, 'lr': 0.0001, 'eps': 1.0, 'init_eps': 1.0,
              'final_eps': 0.0001,'batch_size': 32,'obs': 500,'exp': 5000,
              'fpa': 4,'target_freq': 300,'timestep_train': 10,'max_memory': 500}
    agent = DoubleDQNAgent(state_size, action_size, args.per, params)

    agent.set_model(Networks.dueling_dqn(state_size, action_size, params["lr"]))
    agent.set_target_model(Networks.dueling_dqn(state_size, action_size, params["lr"]))
    
    x_t = game_state.screen_buffer # 480 x 640
    x_t = preprocessImg(x_t, size=(img_rows, img_cols))
    s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    # Start training
    t = 0

    epochs = args.epochs
    games_per_epoch = args.games_per_epoch
    for epoch in range(epochs):
        log_writer.write_log("Epoch: {0}".format(str(epoch)), log_level = LogLevel.Info)
        t = play_epoch(game, agent, epoch, games_per_epoch, t, stats_file_logger)

        # save progress every epoch
        log_writer.write_log("Now we save model", log_level = LogLevel.Info)
        agent.save_model(save_model_file_name)

        # Reset rolling stats buffer
        score_buffer, ammo_buffer, health_buffer = [], [], [] 

        # Write Rolling Statistics to file
        stats_file_logger.writeline("mavg_score: " + str(agent.mavg_score))
        stats_file_logger.writeline("var_score: " + str(agent.var_score))
        stats_file_logger.writeline("mavg_ammo_left: " + str(agent.mavg_ammo_left))
        stats_file_logger.writeline("mavg_kill_counts: " + str(agent.mavg_kill_counts))
        stats_file_logger.writeline("")

    log_writer.write_log("Training done... test time", log_level = LogLevel.Info)
    game = DoomGame()
    game.load_config(args.test_scenario)
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.init()
    action_size = game.get_available_buttons_size()
    episodes_to_watch=args.test_episodes
    for i in range(episodes_to_watch):
        s_t, t, prev_vars = play_episode(game, agent, action_size, 0, i, t, False, delay = 0.1)

        score = game.get_total_reward()    
        log_writer.write_log("Total score: {0}".format(str(score)), log_level = LogLevel.Always)

if __name__ == "__main__":
    main()

