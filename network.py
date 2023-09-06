import numpy as np
import matplotlib.pyplot as plt

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


class NeuralNetwork:
	"""
	Class summary
	"""

	def __init__(self, 
	dims 	= (784, 250, 60, 10), 
	afs 	= ("r", "r", "sm"), 
	classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
	):
		"""
		Construct for a neural network.

		Inputs:
		- dims: a tuple which contains layer sizes in order, of the
		  form (num_inputs, hidden_layer_1, ... , hidden_layer_n, num_classes)
		- afs: a tuple of activation functions for each activating layer, hence
		  of size len(dims) - 1. possible values are:
			- "s" for sigmoid
			- "r" for relu
			- "sm" for softmax (only last layer)
		"""
		# Keep dims and afs
		self.dims = dims
		self.afs = afs

		# Initialising parameters
		self.W = [] # Weight matrices each weight is ~ N(0,1)
		self.b = [] # bias vectors ~N(0,1)
		self.S = [] # S vector used in backpropagation that is, dL/da
		self.n = [] # the unactivated (or normal) output for each layer
		self.a = [] # the activated output for each layer
		self.f = [] # the functions applied at each layer
		self.df = [] # the derivatives of functions applied at each layer

		# Initialise weights, biases, S, n and a.
		for i in range(1, len(dims)):
			self.W.append(np.random.randn(dims[i-1], dims[i])/np.sqrt(dims[i-1]/2))
			self.b.append(np.zeros(dims[i]))
			self.S.append(np.zeros(dims[i]))
			self.n.append(np.zeros(dims[i]))
			self.a.append(np.zeros(dims[i]))
		
		# Get the functions and their derivatives at each layer
		for af in afs:
			if af == "s": # Sigmoid
				self.f.append(np.vectorize(self.sigmoid))
				self.df.append(np.vectorize(self.dsigmoid))
			elif af == "r": # ReLU
				self.f.append(np.vectorize(self.relu))
				self.df.append(np.vectorize(self.drelu))
			elif af == "sm":
				self.f.append(self.softmax)
		
		# Give the class names to the neural network for verbose accuracy
		self.classes = classes

	
	def reset(self):
		# Resets the weights and biases to normally distributed values
		dims = self.dims
		for i in range(1, len(dims)):
			self.W[i-1] = np.random.randn(dims[i-1], dims[i])/np.sqrt(dims[i-1]/2)
			self.b[i-1] = np.zeros(dims[i])
			self.S[i-1] = np.zeros(dims[i])
			self.n[i-1] = np.zeros(dims[i])
			self.a[i-1] = np.zeros(dims[i])
		return

	
	def train(
			self, 
			data		= None, 
			num_epoch	= 5, 
			lr			= 3.4e-3, 
			lr_decay	= 0.25, 
			verbose		= True,
			decay_rate  = "halfway"
			):
		"""
		Inputs:
		
		data: a dictionary of the form:
			data["X"] = matrix of shape (N,D), with N training examples and 
				D dimensions (size of input)
			data["y"] = the corresponding output of shape (N,1), with N 
				true outputs to each X[i]
			note X[i] is the input of the ith example and y[i] is the true
				output

		num_epochs: the number of epochs; times to reuse the whole training 
			data to train

		lr: the learning rate

		lr_decay:  the learning rate decay
		
		decay_rate: string to determine how often to decay.
			"halfway" : 3rd epoch, assumption is that 5 epochs are done,
			"every epoch" : Every epoch.
		"""
		# Get dims and activation functions:
		dims = self.dims
		afs = self.afs

		# Get Weight matrices to not use self all the time
		W = self.W
		b = self.b
		n = self.n
		a = self.a
		S = self.S
		f = self.f
		df = self.df

		# Get the data
		X = data["X"]
		y = data["y"]

		# Get the number of examples
		N = X.shape[0]

		# Keep track of training loss over training
		loss_hist = []
		loss = 0

		# Get the last layer index to make code more understandble
		lli = len(dims) - 2

		# Start training for num_epoch epochs
		for ep in range(num_epoch):
			if verbose:
				print("Epoch: ", ep + 1)
			
			# Randomly shuffle the data
			mask = np.random.choice(N, N, replace=False)
			X = X[mask]
			y = y[mask]

			# Loop over the training set
			for k in range(N):
				
				# First multiplication happens outside because we use X[i] rather than
				# n[i]
				n[0] = X[k] @ W[0] + b[0]
				a[0] = f[0](n[0])

				# Forward propagate and cache values of n (unactivated output at each
				# 	layer) and a (activated output at each layer), note a[i] = f(n[i]).
				for i in range(1, lli + 1):
					n[i] = a[i-1] @ W[i] + b[i]
					a[i] = f[i](n[i])
				
				# Get the true output, in the form of our NN output
				t = np.zeros(10)
				t[y[k]] = 1

				# Calculate error
				e = t - a[lli]				

				# Get loss.
				# Loss will be different for softmax
				if afs[lli] == "sm":
					loss += -np.log(a[lli][y[k]])
				else:
					loss += np.sum(np.abs(e))
				
				# Update loss
				if k % 100 == 0:
					loss_hist.append(loss)
					loss = 0

				# Get the first S valus.
				# We have different cases for if softmax is at the last layer
				if afs[lli] == "sm": # Softmax
					S[lli] = -e
				else:
					A = np.diag(df[lli](n[lli]))
					S[lli] = -2 * A@e
				
				# Get S for the remaining layers, using SAWS
				for i in range(lli, 0, -1):
					A = np.diag(df[i-1](n[i-1]))
					S[i-1] = A @ W[i] @ S[i]
				
				# Update weights and biases.
				# First layer first	
				W[0] = W[0] - lr * np.outer(X[k], (S[0].T)) 
				b[0] = b[0] - lr * S[0]
				
				# Update remaining layers
				for i in range(1, lli + 1):
					W[i] = W[i] - lr * np.outer(a[i-1], (S[i].T)) 
					b[i] = b[i] - lr * S[i]
				
				# Some error checking code
				# if k%200 == 0:
				# 	print(e)
				# 	print("Error sum: ", np.sum(e), "Absolute error sum: ", np.sum(np.abs(e)))
				# 	for i in range(lli+1):
				# 		print(f"Sum of S at layer{i+1}: {np.sum(S[i])}")
					
			# End of one epoch 
			if verbose:
				print("Epoch: ", ep + 1)
				# decay learning rate
				if decay_rate == "halfway":
					if ep == 2:
						print("decayed")
						lr = lr * lr_decay
				elif decay_rate == "every epoch":
					print("decayed")
					lr = lr * lr_decay
		
		# End of all epochs
		# Update model's weights and biases
		self.W = W
		self.b = b

		# return loss_hist, first entry is 0
		return loss_hist[1:]

	
	# function returns % accuracy and TODO outputs confusion matrix if visual=True
	def checkAccuracy(self, data, images=False):
		"""
		Inputs:
			
			data: a dictionary of the form:
				data["X"] = matrix of shape (N,D), with N  examples and 
					D dimensions (size of input)
				data["y"] = the corresponding output of shape (N,1), with N 
					true outputs to each X[i]
				note X[i] is the input of the ith example and y[i] is the true
					output 
			
			images: if True, it prints the true ouput, the predicted output and the
				image; a boolean.
		"""

		# Get class names
		classes = self.classes

		# Get dims and activation functions:
		dims = self.dims

		# Get Weight matrices to not use self all the time
		W = self.W
		b = self.b
		n = self.n
		a = self.a
		f = self.f

		# Get the data
		X = data["X"]
		y = data["y"]

		# Get predicted values for confusion matrix
		y_pred = []

		# Get the number of examples
		N = X.shape[0]

		# Keep track of correct predictions
		correct = 0

		# Loop over the training set
		for k in range(N):

			# Get the last layer index to make code more understandble
			lli = len(dims) - 2
			
			# First multiplication happens outside because we use X[i] rather than
			# n[i]
			n[0] = X[k] @ W[0] + b[0]
			a[0] = f[0](n[0])

			# Forward propagate and cache values of n (unactivated output at each
			# 	layer) and a (activated output at each layer), note a[i] = f(n[i]).
			for i in range(1, lli + 1):
				n[i] = a[i-1] @ W[i] + b[i]
				a[i] = f[i](n[i])
			
			# Get the hottest output 
			pred = np.argmax(a[lli])
			if pred == y[k]:
				correct += 1
			
			# If verbose, print:
			# - The picture
			# - The true output
			# - The neural network output
			if images:
				plt.imshow(X[k].reshape((28,28)))
				plt.title(f"True: {classes[y[k]]} \nPrediction: {classes[pred]}")
				plt.show()
				pass
		
		# return accuracy
		return correct/N
		

	

	# Activation functions and their derivatives
	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))
	
	def dsigmoid(self, x):
		return (1/(1 + np.exp(-x))) * (1 - (1/(1 + np.exp(-x))))

	def relu(self, x):
		return np.maximum(0, x) 
	
	def drelu(self, x):
		return np.maximum(0, x > 0)
	
	def softmax(self, x):
		E = np.exp(x)
		S = np.sum(E)
		return E/S
	

