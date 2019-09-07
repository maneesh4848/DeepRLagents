from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import numpy as np
from tqdm import trange
import argparse
from time import sleep

img_rows, img_cols, img_channels = 64, 64, 4

def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img,size)
    img = skimage.color.rgb2gray(img)

    return img

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--per', action="store_true", help="Use proiritized experience replay")
    parser.add_argument('-s','--scenario',  type=str, required=True, help="Scenario config file")
    parser.add_argument('-m','--model', default="../models/trained_model.h5", type=str, required=False, help="File to save trained model to")
    parser.add_argument('-o','--epochs', default=5, type=int, required=False, help="Number of training epochs")
    parser.add_argument('-g','--games_per_epoch', default=1000, type=int, required=False, help="Number of episodes per epoch")
    parser.add_argument('-t','--test_scenario',  type=str, required=True, help="Scenario config file")
    parser.add_argument('-e','--test_episodes', default=25, type=int, required=False, help="Number of testing episodes")
    parser.add_argument('-l','--log', default="", type=str, required=False, help="File to save logs to")

    return parser

def play_epoch(game, agent, epoch, episodes_per_epoch, t, stats_writer):

    game.new_episode()
    game_state = game.get_state()
    x_t = game_state.screen_buffer # 480 x 640
    x_t = preprocessImg(x_t, size=(img_rows, img_cols))
    s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    action_size = game.get_available_buttons_size()

    # Buffer to compute rolling statistics 
    score_buffer, ammo_buffer, health_buffer, stats_buffer = [], [], [], []

    for g in trange(episodes_per_epoch):
        s_t, t, prev_game_vars = play_episode(game, agent, action_size, epoch, g, t, True, ammo_buffer, health_buffer, score_buffer, stats_buffer, s_t)

        if g%50==0:
            str_buffer = ""
            for tup in stats_buffer:
                str_buffer += str(tup) + '\n'
            stats_writer.write(str_buffer)

    #Save stats_buffer after an epoch also
    str_buffer = ""
    for tup in stats_buffer:
        str_buffer += str(tup) + '\n'
    stats_writer.write(str_buffer)

    # Save Agent's Performance Statistics, every epoch
    print("Update Rolling Statistics")
    agent.mavg_score.append(np.mean(np.array(score_buffer)))
    agent.var_score.append(np.var(np.array(score_buffer)))
    agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
    agent.mavg_kill_counts.append(np.mean(np.array(health_buffer)))

    return t

def play_episode(game, agent, action_size, epoch, game_num, t, learn = True, ammo_buffer = None, health_buffer = None, score_buffer = None, stats_buffer = None, s_t = None, delay = 0):

    game.new_episode()
    game_state = game.get_state()
    game_vars = game_state.game_variables
    prev_game_vars = game_vars

    if s_t is None:
        x_t = game_state.screen_buffer # 480 x 640
        x_t = preprocessImg(x_t, size=(img_rows, img_cols))
        s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
        s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    while not game.is_episode_finished():
        if learn:
            loss = 0
            Q_max = 0

        a_t = np.zeros([action_size])

        # Epsilon Greedy
        action_idx  = agent.get_action(s_t)
        a_t[action_idx] = 1

        a_t = a_t.astype(int)
        game.set_action(a_t.tolist())
        skiprate = agent.get_frameskip_rate()
        game.advance_action(skiprate)

        r_t = game.get_last_reward()  #each frame we get reward of 0.1, so 4 frames will be 0.4

        if learn:
            game_state = game.get_state()  # Observe again after we take the action
            is_terminated = game.is_episode_finished()

            # Log stats
            if is_terminated:
                ammo_buffer.append(game_vars[0])
                health_buffer.append(game_vars[1])
                score = game.get_total_reward()
                score_buffer.append(score)

                stats_buffer.append((epoch,game_num,game_vars[0],game_vars[1],score)) #For printing to file
                print ("Episode Finish ", game_vars, score)

                break

        x_t1 = game_state.screen_buffer
        game_vars = game_state.game_variables

        x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)                
        r_t = agent.shape_reward(r_t, game_vars, prev_game_vars)

        # Update the cache
        prev_game_vars = game_vars

        if learn:
            # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
            agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)

            # Do the training (if needed)
            new_Q_max, new_loss = agent.train_replay(t)
            if new_Q_max != None and new_loss != None:
                Q_max, loss = new_Q_max, new_loss

        s_t = s_t1
        t += 1

        # print info
        #state = ""
        #if t <= agent.observe:
        #    state = "observe"
        #elif t > agent.observe and t <= agent.observe + agent.explore:
        #    state = "explore"
        #else:
        #    state = "train"

        #if (is_terminated):
        #    print("TIME", t, "/ STATE", state, \
        #          "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
        #         "/ Q_MAX %e" % np.max(Q_max), "/ LOSS", loss)

        #train_scores = np.array(score_buffer)
        #sleep(0.1)

        #print("Results: mean: %.1f-%.1f," % (train_scores.mean(), train_scores.std()), \
                #  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        if delay > 0:
            sleep(delay)

    return s_t, t, prev_game_vars

