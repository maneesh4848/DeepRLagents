from abc import ABCMeta, abstractmethod
from collections import deque
from builtins import *


def sanity_check_params(params, default_params):
    for param in default_params.keys():
        if param not in params:
            params[param] = default_params[param]

class Network_Base:

    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.__name = name
        super().__init__()

        # create model
        self._model = None

    @abstractmethod
    def get_action(self, state):
        """
        Get action from model given the current state
        """
        pass

    @abstractmethod
    def load_model(self, filename):
        """
        Load model from file
        """
        pass

    @abstractmethod
    def save_model(self, filename):
        """
        Save model to file
        """
        pass

class Network(Network_Base):

    def __init__(self, name, state_size, action_size, params):

        default_params = {'gamma': 0.99, 'lr': 0.0001, 'eps': 1.0, 'fpa': 4, 'max_memory': 500}

        sanity_check_params(params, default_params)
        super().__init__(name)

        # get size of state and action
        self._state_size = state_size
        self._action_size = action_size

        # create replay memory using deque
        self._memory = deque()
        self._max_memory = params["max_memory"] # number of previous transitions to remember

        # setting hyperparameters
        self._gamma = params["gamma"]
        self._learning_rate = params["lr"]
        self._epsilon = params["eps"]
        self._frame_per_action = params['fpa']

        # Performance Statistics
        self.stats_window_size = 50 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time
        self.mavg_ammo_left = [] # Moving Average of Ammo used
        self.mavg_kill_counts = [] # Moving Average of Kill Counts


    def set_model(self, model):
        self._model = model

    def get_action(self, state):
        q = self._model.predict(state)
        action_idx = np.argmax(q)
        return action_idx

    def get_frameskip_rate(self):
        return self._frame_per_action

    @abstractmethod
    def train_replay(self, t):
        """
        Train model using random samples from replay memory
        """
        pass

    def load_model(self, filename):
        self._model.load_weights(name)

    def save_model(self, name):
        self._model.save_weights(name, overwrite=True)

