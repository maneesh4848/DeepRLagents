from keras.models import Sequential
from keras import layers as kl
from keras.initializers import Constant

class DDQN:
	pass

class DQN:
	
	def __init__(self, state_space, action_space):
		self.model = Sequential()
		self.create_model(state_space, action_space)

	def create_model(self, input_dim, output_dim):
		self.model.add(kl.Conv2D(16, (4,4), (4,4), activation = 'relu', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = Constant(value = 0.1), input_shape = input_dim))
		self.model.add(kl.Conv2D(32, (2,2), (2,2), activation = 'relu', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = Constant(value = 0.1), input_shape = input_dim))
		self.model.add(kl.Flatten())
		self.model.add(kl.Dense(1024, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = Constant(value = 0.1)))
		self.model.add(kl.Dense(output_dim, use_bias = True, kernel_initializer = 'he_normal', bias_initializer = Constant(value = 0.1)))

def main():
    pass
if __name__ =- '__main__':
    main()
