#!/usr/bin/env python
from __future__ import print_function
import pdb
import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time
from datetime import datetime
from tqdm import trange

import os

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import itertools as it
from time import sleep
import tensorflow as tf
import time
from keras_networks import Networks


def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img,size)
    img = skimage.color.rgb2gray(img)

    return img
    

class DoubleDQNAgent:

    def __init__(self, state_size, action_size):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 500
        self.explore = 5000 
        self.frame_per_action = 4
        self.update_target_freq = 300 
        self.timestep_per_train = 10 # Number of timesteps between training interval

        # create replay memory using deque
        self.memory = deque()
        self.max_memory = 5000 # number of previous transitions to remember

        # create main model and target model
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.stats_window_size= 50 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time
        self.mavg_ammo_left = [] # Moving Average of Ammo used
        self.mavg_kill_counts = [] # Moving Average of Kill Counts

    def set_learning_rate(self,l_rate=0.0001):
        self.learning_rate=l_rate

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            q = self.model.predict(state)
            action_idx = np.argmax(q)
        return action_idx

    def shape_reward(self, r_t, misc, prev_misc, t):
        
        
        if (misc[0] > prev_misc[0]): # Use ammo
            r_t = r_t - 0.08

        if (misc[1] < prev_misc[1]): # LOSS HEALTH
            r_t = r_t - 0.1

        #if (misc[2] < prev_misc[2]): # KILL COUNT. No extra reward other than what games gives
        #    r_t = r_t - 0.1

        return r_t

    # Save trajectory sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        # Update the target model to be same with model
        if t % self.update_target_freq == 0:
            self.update_target_model()

    # Pick samples randomly from replay memory (with batch_size)
    def train_minibatch_replay(self):
        """
        Train on a single minibatch
        """
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros(((batch_size,) + self.state_size)) # Shape 64, img_rows, img_cols, 4
        update_target = np.zeros(((batch_size,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i,:,:,:] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i,:,:,:] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input) # Shape 64, Num_Actions

        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        #print ("HERE!!!!",update_input.shape)
        loss = self.model.train_on_batch(update_input, target)

        return np.max(target[-1]), loss

    # Pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        
        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        #print ("Check!!",)
        replay_samples = random.sample(self.memory, num_samples)
        #pdb.set_trace()
        update_input = np.zeros(((num_samples,) + self.state_size)) 
        update_target = np.zeros(((num_samples,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(num_samples):
            update_input[i,:,:,:] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            update_target[i,:,:,:] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        target = self.model.predict(update_input) 
        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(num_samples):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

        #print ("HERE!!!!",update_input.shape)
        loss = self.model.fit(update_input, target, batch_size=self.batch_size, nb_epoch=1, verbose=0)

        return np.max(target[-1]), loss.history['loss']

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config("./scenarios/simpler_basic.cfg")
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.init()

    json_file_name = 'statistics/data_test_del_me.txt' #Saved every episode # (epoch_num,game_num,ammo,health,score)
    load_model_file_name = "./models/new_ddqn_basic_ph1.h5" #Model from which it is initialized
    save_model_file_name = "models/ddqn_test_del_me.h5" #Saved every epoch
    save_stats_file_name = "statistics/ddqn_test_del_me.txt" #Saved every epoch

    try:
        os.remove(json_file_name) #Remove the json if initially present because it appends!!
    except OSError:
        pass


    game_state = game.get_state()
    #misc = game_state.game_variables 
    misc = game_state.game_variables  # [AMMO, HEALTH, KILLCOUNT]
    print ("HERE@@@@@@@@@@@: "+str(misc))
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows , img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 4 # We stack 4 frames

    state_size = (img_rows, img_cols, img_channels)
    agent = DoubleDQNAgent(state_size, action_size)

    agent.model = Networks.new_dqn(state_size, action_size, agent.learning_rate)
    agent.target_model = Networks.new_dqn(state_size, action_size, agent.learning_rate)
    
    agent.load_model(load_model_file_name)
    agent.update_target_model()
    """
    end_layer=4 #Change the number accordingly #Here leaves Dense, BatchNorm, ReLU, Dense of PG
    for layer in agent.model.layers[:end_layer]: 
        layer.trainable=False
    for layer in agent.model.layers[end_layer:]:
        layer.trainable=True

    for layer in agent.target_model.layers[:end_layer]: 
        layer.trainable=False
    for layer in agent.target_model.layers[end_layer:]:
        layer.trainable=True

    agent.set_learning_rate(0.0001) #IMP! #While fine-tuning need lower learning rate. Here left same
    """
    x_t = game_state.screen_buffer # 480 x 640
    x_t = preprocessImg(x_t, size=(img_rows, img_cols))
    s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    # Start training
    epsilon = agent.initial_epsilon
    #GAME = 0
    t = 0
    #max_life = 0 # Maximum episode life (Proxy for agent performance)
    #life = 0

    # Buffer to compute rolling statistics 
    score_buffer, ammo_buffer, health_buffer = [], [], [] 
    
    #pdb.set_trace()
    #while (not game.is_episode_finished()) and (t<100):
    epochs = 5
    games_per_epoch = 10
    json_data={}
    temp_buffer=[]
    for ep in range(epochs):
        print("Epoch:", ep)
        #is_terminated = game.is_episode_finished()
        for g in trange(games_per_epoch,leave=False):
            game.new_episode()
            game_state = game.get_state()
            misc = game_state.game_variables
            x_t1 = game_state.screen_buffer
            x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
            x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            #num_kills=0 #Just for json_file printing
            while not game.is_episode_finished():
                #pdb.set_trace()
                #print ("GOING!!!!")
                loss = 0
                Q_max = 0
                r_t = 0
                a_t = np.zeros([action_size])

                # Epsilon Greedy
                action_idx  = agent.get_action(s_t)
                a_t[action_idx] = 1

                a_t = a_t.astype(int)
                game.set_action(a_t.tolist())
                skiprate = agent.frame_per_action
                game.advance_action(skiprate)

                game_state = game.get_state()  # Observe again after we take the action
                is_terminated = game.is_episode_finished()

                r_t = game.get_last_reward()  #each frame we get reward of 0.1, so 4 frames will be 0.4

                if (is_terminated):
                    #if (life > max_life):
                    #    max_life = life
                    #GAME += 1
                    #life_buffer.append(life)
                    ammo_buffer.append(misc[0])
                    health_buffer.append(misc[1])
                    score = game.get_total_reward()
                    score_buffer.append(score)

                    temp_buffer.append((ep,g,misc[0],misc[1],score)) #For printing to file
                    print ("Episode Finish ", misc, score)
        
                    break
                    #game.new_episode()
                    
                x_t1 = game_state.screen_buffer
                misc = game_state.game_variables

                x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
                x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
                s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)                
                r_t = agent.shape_reward(r_t, misc, prev_misc, t)

                # Update the cache
                prev_misc = misc

                # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
                agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)

                # Do the training
                if t > agent.observe and t % agent.timestep_per_train == 0:
                    #pdb.set_trace()
                    Q_max, loss = agent.train_replay()
                    
                s_t = s_t1
                t += 1                    

                # print info
                state = ""
                if t <= agent.observe:
                    state = "observe"
                elif t > agent.observe and t <= agent.observe + agent.explore:
                    state = "explore"
                else:
                    state = "train"

                if (is_terminated):
                    print("TIME", t, "/ STATE", state, \
                          "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
                          "/ Q_MAX %e" % np.max(Q_max), "/ LOSS", loss)

                train_scores = np.array(score_buffer)
                sleep(0.1)

                #print("Results: mean: %.1f-%.1f," % (train_scores.mean(), train_scores.std()), \
                #  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            if g%50==0:
                with open(json_file_name,'a') as myfile:
                    for tup in temp_buffer:
                        myfile.write(str(tup)+'\n')
                    temp_buffer=[]

        #Save temp_buffer after an epoch also
        with open(json_file_name,'a') as myfile:
            for tup in temp_buffer:
                myfile.write(str(tup)+'\n')
            temp_buffer=[]

        # save progress every epoch
        print("Now we save model")
        agent.model.save_weights(save_model_file_name, overwrite=True)

        # Save Agent's Performance Statistics, every epoch
        print("Update Rolling Statistics")
        agent.mavg_score.append(np.mean(np.array(score_buffer)))
        agent.var_score.append(np.var(np.array(score_buffer)))
        agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
        agent.mavg_kill_counts.append(np.mean(np.array(health_buffer)))

        # Reset rolling stats buffer
        life_buffer, ammo_buffer, health_buffer = [], [], [] 

        # Write Rolling Statistics to file
        with open(save_stats_file_name, "w") as stats_file:
            #stats_file.write('Game: ' + str(GAME) + '\n')
            #stats_file.write('Max Score: ' + str(max_life) + '\n')
            stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
            stats_file.write('var_score: ' + str(agent.var_score) + '\n')
            stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
            stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')
            #pdb.set_trace()

    print("Training done.... test time")
    game = DoomGame()
    game.load_config("./scenarios/simpler_basic.cfg")
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.init()
    action_size = game.get_available_buttons_size()
    episodes_to_watch=10
    for i in range(episodes_to_watch):
        game.new_episode()
        game_state = game.get_state()
        misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
        prev_misc = misc
        x_t = game_state.screen_buffer # 480 x 640
        x_t = preprocessImg(x_t, size=(img_rows, img_cols))
        s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
        s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

        while not game.is_episode_finished():

            a_t = np.zeros([action_size])
            
            action_idx  = agent.get_action(s_t)
            a_t[action_idx] = 1

            a_t = a_t.astype(int)
            game.set_action(a_t.tolist())
            skiprate = agent.frame_per_action
            game.advance_action(skiprate)

            #for j in range(frame_per_action):
            #    game.advance_action()

            x_t1 = game_state.screen_buffer
            misc = game_state.game_variables

            x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
            x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            r_t = agent.shape_reward(r_t, misc, prev_misc, t)
            s_t=s_t1
            x_t=x_t1

            sleep(0.5)
        score = game.get_total_reward()    
        print("Total score: ",score)



