"""RecurrentNetworkEntropy.py
~~~~~~~~~~~~~~
A simple recurrent neural network, implementing stochastic gradient 
descent learning algorithm. The network is composed of 1 (one) hidden
layer, the neurons activate using the tanh function, and we use the
cross-entropy cost function to evaluate accuracy.  
"""

# TODO: l_1 Regularization 

# TODO: Early Stopping

#### Libraries ####
# Standard Libraries
import json
from random import shuffle

# Third Party Libraries
import numpy as np 

# User Libraries
from Text_Vec import Text_Vec

__version__ = "0.9"
__author__ = "Lukasz Obara"

# Misc functions
def softmax(z):
	""" The softmax function.
	"""
	return np.exp(z) / np.sum(np.exp(z))

class RNN(object):
	def __init__(self, data, hidden_size, spec_rad = 1.01):
		self.data = data
		self.hidden_size = hidden_size 
		self.weights_hidden = np.random.rand(hidden_size, hidden_size) * 0.1 # W
		self.weights_input = np.random.rand(hidden_size, len(data[0])) * 0.1 # U
		self.weights_output = np.random.rand(len(data[0]), hidden_size) * 0.1 # V 
		self.bias_hidden = np.array([np.random.rand(hidden_size)]).T  # b
		self.bias_output = np.array([np.random.rand(len(data[0]))]).T # c
		# Initialization of hidden state for time 0. 
		self.h_0 = np.array([np.zeros(hidden_size)]).T

		# Readjustment of spectral radius
		largest_eig = max(abs(np.linalg.eigvals(self.weights_hidden)))
		self.weights_hidden = self.weights_hidden / largest_eig
		self.weights_hidden = spec_rad * self.weights_hidden

		# Initial cache values for update of RMSProp
		self.cache_w_hid = np.zeros((hidden_size, hidden_size))  
		self.cache_w_in = np.zeros((hidden_size, len(data[0])))
		self.cache_w_out = np.zeros((len(data[0]), hidden_size))
		self.cache_b_hid = np.zeros((hidden_size, 1))
		self.cache_b_out = np.zeros((len(data[0]), 1))

	def train(self, seq_length, epochs, learning_rate=0.001, decay_rate=0.9, 
			  eps=0.0001, annealing_rate=0, randomize=False, print_final=True):
	
		accuracy, evaluation_cost = [], []

		sequences = [self.data[i:i+seq_length] \
					 for i in range(0, len(self.data), seq_length)] 

		for epoch in range(epochs):
			if randomize:
				shuffle(sequences)
				
			print('epoch {}'.format(epoch))
			accu = 0
			loss = 0

			for seq in sequences:				
				for s in seq:
					self.update(seq, epoch, learning_rate, decay_rate, 
								learning_decay)

					_, outputs, loss = self.feedforward(seq)
			
					final_text =  chr(np.argmax(seq))

					for j in range(len(outputs)):
						num = np.argmax(outputs[j])
						final_text += chr(num)

					if num == np.argmax(seq[j+1]):
						accu += 1

				if print_final:
					print(final_text)

				loss += loss 

			accuracy.append(accu)
			evaluation_cost.append(loss)

			print("The loss at epoch {} is: {}".format(epoch, loss))
			print("The accuracy is {} / {}".format(accu, len(self.data)))
			print('---------------')
 
	def update(self, seq, epoch, learning_rate, decay_rate, annealing_rate, 
			   eps):
		"""
		Updates the network's weights and biases by applying gradient
		descent using backpropagation through time and RMSPROP. 
		"""
		def update_rule(cache_attr, x_attr, dx):
			cache = getattr(self, cache_attr)
			cache = decay_rate * cache + (1 - decay_rate) * dx**2
			setattr(self, cache_attr, cache)

			x = getattr(self, x_attr)
			x -= (eta * learning_rate) * dx / (np.sqrt(cache) + eps)
			setattr(self, x_attr, x)

		eta = np.exp(-annealing_rate*epoch)

		delta_nabla_c, delta_nabla_b,\
		delta_nabla_V, delta_nabla_W, delta_nabla_U = self.backward_pass(seq)

		update_rule('cache_w_hid', 'weights_hidden', delta_nabla_W)
		update_rule('cache_w_in', 'weights_input', delta_nabla_U)
		update_rule('cache_w_out', 'weights_output', delta_nabla_V)
		update_rule('cache_b_hid', 'bias_hidden', delta_nabla_b)
		update_rule('cache_b_out', 'bias_output', delta_nabla_c)

	def feedforward(self, sequence):
		"""
		Returns a tuple `(hidden_states, output_states)` representing
		the hidden and output states at each time step
		"""
		total_loss = 0
		hidden_states = [self.h_0]
		output_states = []

		for t in range(len(sequence)-1):
			# Since `hidden_states` includes the initialization it is 
			# shifted to the right by 1 time step allowing us to couple
			# the input and hidden state by `t` and not worry about 
			# `t-1` for the hidden state as found in the notes.

			activation = np.dot(self.weights_input, sequence[t]) \
						+np.dot(self.weights_hidden, hidden_states[t]) \
						+self.bias_hidden
			h = np.tanh(activation)
			o = np.dot(self.weights_output, h) + self.bias_output

			hidden_states.append(h)
			output_states.append(o)

			total_loss += self.loss(softmax(o), sequence[t]+1)

		return (hidden_states, output_states, total_loss)

	def backward_pass(self, sequence):
		"""
		Return a tuple `(nabla_c, nabla_b, nabla_V, nabla_W, nabla_U)
		represeting the gradient for the cross entropy cost function. 
		Each element in the tuple is a numpy array.  
		"""
		h_states, o_states, _ = self.feedforward(sequence)
		# The last time step
		nabla_o = self.d_loss(softmax(o_states[-1]),sequence[-1])
		nabla_c = nabla_o
		nabla_V = np.dot(nabla_o, h_states[-1].T)

		# `nabla_h L` for the last time step.
		nabla_h = np.dot(self.weights_output.T, nabla_o)

		# Preliminary computations needed for hidden nodes
		nabla_temp = np.multiply(1-h_states[-1]**2, nabla_h)

		nabla_b = nabla_temp
		nabla_W = np.dot(nabla_temp, h_states[-2].T)
		nabla_U = np.dot(nabla_temp, sequence[-2].T)

		# `h_states` includes `h_0` hence making h_states of length 
		# `len()+1`. As such, we need to to compensate for this when 
		# considering the current time state we must add by 1.

		for t in reversed(range(1, len(sequence)-1)):
			# We start by computing `nabla_o` for the second to last 
			# step and apply the backprop through time formulas
			nabla_o = self.d_loss(softmax(o_states[t-1]), sequence[t])
			
			# \nabla_c L
			nabla_c += nabla_o

			# \nabla_V L
			nabla_V += np.dot(nabla_o, h_states[t].T) 

			# \nabla_h L
			diag_nabla_h = np.diag(h_states[t+1][:,0])
			diag_ht = np.identity(self.hidden_size) - diag_nabla_h**2
			nabla_h_temp = np.dot(self.weights_hidden.T, diag_ht)
			nabla_h = np.dot(nabla_h_temp, nabla_h) \
					+ np.dot(self.weights_output.T, nabla_o)

			nabla_temp = np.multiply(1-h_states[t]**2, nabla_h)

			nabla_b += nabla_temp
			nabla_W += np.dot(nabla_temp, h_states[t-1].T)
			nabla_U += np.dot(nabla_temp, sequence[t-1].T)

		# clip to mitigate exploding gradients
		for delta in [nabla_c, nabla_b, nabla_V, nabla_W, nabla_U]:
			np.clip(delta, -3, 3, out=delta)

		return (nabla_c, nabla_b, nabla_V, nabla_W, nabla_U)

	def loss(self, output_activation, y):
		"""Returns the cross-entropy loss.
		"""

		return -np.max(np.multiply(np.log(output_activation), y))

	def d_loss(self, output_activation, y):
		"""Returns the vector of partial derivative \nabla_{o^{(t)}}L
		for the output activations of the cross entropy loss.
		"""
		
		return (output_activation-y)

	def save(self, filename):
		"""
		Save the neural network to the file ``filename``.
		"""
		data = {'sizes': self.sizes,
				'weights_inp': [w.tolist() for w in self.weights_input],
				'weights_hid': [w.tolist() for w in self.weights_hidden],
				'weights_out': [w.tolist() for w in self.weights_output],
				'biases_hid': [b.tolist() for b in self.bias_hidden],
				'biases_out': [b.tolist() for b in self.bias_output]}
		f = open(filename, "w")
		json.dump(data, f)
		f.close()

if __name__ == '__main__':
	pass
