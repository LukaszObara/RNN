"""RecurrentNetwork.py
~~~~~~~~~~~~~~
A very simple recurrent neural netowrk based of network.py, implementing 
the stochastic gradient descent learning algorithm. The network is 
composed of 1 (one) hidden layer and the activation neurons use a 
sigmoid function and the quadratic cost function is used to evaluate 
performance.  
"""

# TODO: regularization to cost

#### Libraries ####
# Third Party Libraries
import numpy as np 

__version__ = "0.01"
__author__ = "Lukasz Obara"

# Misc functions 
# def softmax(z):
# 	""" The softmax function.
# 	"""
# 	return np.exp(z) / np.sum(np.exp(z))

def sigmoid(z):
	""" The sigmoid function used in activating the neurons. 
	"""
	return 1.0/(1.0+np.exp(-z))

def d_sigmoid(z):
	""" The derivative of the sigmoid function.
	"""
	return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
	""" The tanh function used in activating the neurons. 
	"""
	return np.tanh(z)

def d_tanh(z):
	""" The derivative of the tanh function.
	"""
	return (1-tanh(z)**2) 

class RNN(object):
	def __init__(self, data, hidden_size):
		self.data = data
		self.time_periods = len(data)
		# Initializes the weights, W, U, and V from the notes. 
		self.hidden_weights = np.random.rand(hidden_size, hidden_size) # W
		self.input_weights = np.random.rand(hidden_size, len(data[0][0])) # U
		self.output_weights = np.random.rand(len(data[0][1]), hidden_size) # V
		# Sets the biases for both the neurons and output neuron. That 
		# is "b" and "c" from the notes.  
		self.hidden_bias = np.array([np.random.rand(hidden_size)]).T # b
		self.output_bias = np.array([np.random.rand(len(data[0][1]))]) # c
		# Initialization of hidden state for time 0. 
		self.h_0 = np.array([np.zeros(hidden_size)]).T

	def update_batch(self):
		pass

	def feedforward(self):
		# Outputs the first activation and hidden state h^{(1)}
		activation = np.dot(self.input_weights, self.data[0][0]) \
					+np.dot(self.hidden_weights, self.h_0) \
					+self.hidden_bias
		h = sigmoid(activation)
		# Outputs the first output state o^{(1)}
		o = np.dot(self.output_weights, self.h_0) + self.output_bias
		hidden_states = [self.h_0, h]
		output_states = [o]

		for t in range(1, self.time_periods):
			# The value h is of the previous state and is alreay store 
			# in memory, it gets changed at every iteration. This is the 
			# reason we see h in activation and not hidden_states[t-1].
			activation = np.dot(self.input_weights, self.data[t][0]) \
						+np.dot(self.hidden_weights, h) \
						+self.hidden_bias
			h = sigmoid(activation)
			o = np.dot(self.output_weights, h) + self.output_bias

			hidden_states.append(h)
			output_states.append(o)

		return (hidden_states, output_states)

	def backward_pass(self):
		h_states, o_states = self.feedforward()

		# \nabla_c L = sum_t \nabla_{o^{(t)}}L
		nabla_c = 0 
		for t in range(self.time_periods):
			nabla_c += self.d_loss(sigmoid(o_states[t]), self.data[t][1]) \
					  *d_sigmoid(o_states[t])

		# \nabla_V L = \sum_t \nabla_{o^{(t)}}L * transpose(h^{(t)})
		nabla_V = 0
		for t in range(self.time_periods):
			nabla_o = self.d_loss(sigmoid(o_states[t]), self.data[t][1]) \
				 	 *d_sigmoid(o_states[t])
			nabla_V += np.dot(nabla_o, h_states[t].T)

		# \nabla_b L = \sum_t diag(1-(h^{(t)})^2) * \nabla_{h^{(t)}} L 
		# \nabla_W L = \sum_t diag(1-(h^{(t)})^2) * \nabla_{h^{(t)}} L * h^{(t)}^T
		# \nabla_U L = \sum_t diag(1-(h^{(t)})^2) * \nabla_{h^{(t)}} L * x^{(t)}^T      
		# Computes nabla_h for the last time step.
		nabla_h = np.dot(self.output_weights.T, 
						  self.d_loss(o_states[-1], 
						 	self.data[-1][1])*d_sigmoid(o_states[-1]))

		diag = h_states[-1] - h_states[-1]**2 
		diag = np.diag(diag[:,0])
		
		# h_states includes h_0 hence making h_states of lenght 12+1. As
		# such, we need to to compensate for this when considering the 
		# time steps t-1, by either substracting by 1 when considering 
		# the previous state, or adding by 1 when considering the cur-
		# rent state. 

		# nabla_b, nabla_W, nabla_U for last time step.
		nabla_b = np.dot(diag, nabla_h)
		nabla_W = np.dot(nabla_b, h_states[-2].T)
		nabla_U = np.dot(nabla_b, self.data[-1][0].T)

		for t in reversed(range(self.time_periods-1)):
			diag = h_states[t+2] - h_states[t+2]**2
			diag = np.diag(diag[:,0])

			h_hor = np.dot(diag, np.dot(self.hidden_weights.T, nabla_h))
			h_ver = np.dot(self.output_weights.T, 
						    self.d_loss(o_states[t], self.data[t][1])\
						   *d_sigmoid(o_states[t]))
			nabla_h = h_hor + h_ver

			diag_cur = h_states[t+1] - h_states[t+1]**2
			diag_cur = np.diag(diag_cur[:,0])

			nabla_b += np.dot(diag_cur, nabla_h)
			nabla_W += np.dot(diag_cur, np.dot(nabla_h, h_states[t+1].T))
			nabla_U += np.dot(diag_cur, np.dot(nabla_h, self.data[t][0].T))
		
		return (nabla_c, nabla_b, nabla_V, nabla_W, nabla_U)

	def d_loss(self, output_activation, y):
		"""Returns the vector of partial derivative \nabla_{o^{(t)}}L
		for the output activations of the quadratic loss."""
		return (output_activation-y)


if __name__ == '__main__':
	# Used for test data creation 
	# f = lambda a, b, c, d: a**(-3) * c - np.sin(b+a/(b+1)) + 0.4

	# for i in range(1, len(x)):
	# 	previous = f(x[i-1][0][0], x[i-1][0][1], x[i-1][0][2], x[i-1][0][3])
	# 	current = f(x[i][0][0], x[i][0][1], x[i][0][2], x[i][0][3])
	# 	current = current * np.cos(previous+current)
	# 	if current > 0:
	# 		print(1)
	# 	else:
	# 		print(0)

	x = [(np.array([[0.123], [0.023], [0.000], [0.123]]), np.array([[0]])), 
		 (np.array([[0.001], [0.031], [0.045], [0.621]]), np.array([[0]])),
		 (np.array([[0.312], [0.032], [0.012], [0.009]]), np.array([[0]])),
		 (np.array([[0.423], [0.012], [0.314], [0.943]]), np.array([[0]])),
		 (np.array([[0.532], [0.000], [0.000], [0.642]]), np.array([[1]])),
		 (np.array([[0.631], [0.000], [0.329], [0.981]]), np.array([[1]])),
		 (np.array([[0.719], [0.004], [0.111], [0.700]]), np.array([[1]])),
		 (np.array([[0.848], [0.515], [0.022], [0.032]]), np.array([[0]])),
		 (np.array([[0.900], [0.624], [0.211], [0.001]]), np.array([[0]])),
		 (np.array([[0.100], [0.215], [0.000], [0.023]]), np.array([[1]])),
		 (np.array([[0.010], [0.524], [0.005], [0.238]]), np.array([[1]])),
		 (np.array([[0.120], [0.321], [0.001], [0.981]]), np.array([[1]]))]

	# Used for test data 
	# for i in range(1, len(x)):
	# 	previous = f(x[i-1][0][0], x[i-1][0][1], x[i-1][0][2], x[i-1][0][3])
	# 	current = f(x[i][0][0], x[i][0][1], x[i][0][2], x[i][0][3])
	# 	current = current * np.cos(previous+current)
	# 	if current > 0:
	# 		print(1)
	# 	else:
	# 		print(0)

	test = [(np.array([[0.75], [0.75], [0.5]]), np.array([[0.50]])), 
			(np.array([[0.50], [0.02], [1.0]]), np.array([[0.00]])),
			(np.array([[1.00], [0.54], [0.5]]), np.array([[0.75]])),
			(np.array([[0.00], [0.12], [0.0]]), np.array([[0.00]]))]

	rnn = RNN(x, 5)
	# print(rnn.hidden_bias)
	rnn.backward_pass()
	# print(rnn.data[0][1])
	# print(np.dot(rnn.output_weights,rnn.h_0))