import random
from random import choice
import numpy as np
from network import sanity_check_params
from network import Network
from builtins import super

class DoubleDQNAgent(Network):

    def __init__(self, state_size, action_size, use_per, params):

        default_params = {'gamma': 0.99, 'lr': 0.0001, 'eps': 1.0, 'init_eps': 1.0,
                          'final_eps': 0.0001,'batch_size': 32,'obs': 500,'exp': 5000,
                          'fpa': 4,'target_freq': 300,'timestep_train': 10,'max_memory': 500, 'per_const': 0.5}

        sanity_check_params(params, default_params)
        super().__init__("DoubleDQN", state_size, action_size, use_per, params)

        # these is hyper parameters for the Double DQN
        self._initial_epsilon = params['init_eps']
        self._final_epsilon = params['final_eps']
        self._batch_size = params['batch_size']
        self._observe = params['obs']
        self._explore = params['exp']
        self._update_target_freq = params['target_freq']
        self._timestep_per_train = params['timestep_train'] # Number of timesteps between training interval

        self._target_model = None

    def set_target_model(self, model):
        """
        Set target model
        """
        self._target_model = model

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self._target_model.set_weights(self._model.get_weights())

    def shape_reward(self, r_t, misc, prev_misc):
        if (misc[0] > prev_misc[0]): # Use ammo
            r_t = r_t - 0.08

        if (misc[1] < prev_misc[1]): # LOSS HEALTH
            r_t = r_t - 0.1
        return r_t

    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        """
        Save trajectory sample <s,a,r,s'> to the replay memory
        """
        self._memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        if self._use_per:
	    prior=(self.eval_replay(self._memory[-1])+1e-8)**self._prioritization_const
	    self._priorities.append(prior)
        if self._epsilon > self._final_epsilon and t > self._observe:
            self._epsilon -= (self._initial_epsilon - self._final_epsilon) / self._explore

        if len(self._memory) > self._max_memory:
            self._memory.popleft()
            if self._use_per:
	        self._priorities.popleft()

        # Update the target model to be same with model
        if t % self._update_target_freq == 0:
            self.update_target_model()

    def eval_replay(self, replay):
        """
        Used by PER, evaluates replay and returns loss
        """		
        eval_input = np.zeros(((1,) + self._state_size))
        eval_target = np.zeros(((1,) + self._state_size))
        
        eval_input[0,:,:,:] = replay[0]
        action = replay[1]
        reward = replay[2]
        eval_target[0,:,:,:] = replay[3]
        done = replay[4]
        
        target = self._model.predict(eval_input) # Q(s) = [for all a, Q(s,a)]
        target_val = self._model.predict(eval_target) # Q(s') = [for all a, Q(s',a)]
        target_val_ = self._target_model.predict(eval_target) # Q'(s') = [for all a, Q'(s',a)]

        old_q = target[0][action]
        # like Q Learning, get maximum Q value at s'
        # But from target model
        if done:
                target[0][action] = reward # final action leads to reward
        else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[0]) # best action
                target[0][action] = reward + self._gamma * (target_val_[0][a]) # DDQN update rule

        loss = abs(old_q - target[0][action])

        return loss

    def get_action(self, state):
        if np.random.rand() <= self._epsilon:
            action_idx = random.randrange(self._action_size)
        else:
            q = self._model.predict(state)
            action_idx = np.argmax(q)
        return action_idx

    def train_replay(self, t):
        if t > self._observe and t % self._timestep_per_train == 0:
            return self.train()
        return None, None

    def train(self):
        num_samples = min(self._batch_size * self._timestep_per_train, len(self._memory))
	replay_samples=[]
        if self._use_per:
	    weights=np.array(self.priorities)
	    weights=weights*1.0/weights.sum(0)
	    replay_indices = np.random.choice(len(self.memory), size=num_samples, p=weights)
	    for idx in replay_indices:
		replay_samples.append(self.memory[idx])
        else:
            replay_samples = random.sample(self._memory, num_samples)
        update_input = np.zeros(((num_samples,) + self._state_size)) 
        update_target = np.zeros(((num_samples,) + self._state_size))
        action, reward, done = [], [], []

        for i in range(num_samples):
            update_input[i,:,:,:] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            update_target[i,:,:,:] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        target = self._model.predict(update_input) 
        target_val = self._model.predict(update_target)
        target_val_ = self._target_model.predict(update_target)

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
                target[i][action[i]] = reward[i] + self._gamma * (target_val_[i][a])

        loss = self._model.fit(update_input, target, batch_size=self._batch_size, epochs=1, verbose=0)

        return np.max(target[-1]), loss.history['loss']

