import numpy as np

class MLP_Layer:
	"""One layer of an MLP. """


	def __init__(self, input_dim:int, output_dim:int, activation_function:str, seed:int=42):
		"""Initialize the MLP layer. 
		
		Args:
			input_dim (int): Dimension of input. 
			output_dim (int): Dimension of output. 
			activation_function (str): The activation function of the layer. Can be ReLU, softmax.
			seed (int): The seed for the random number generator. """
		
		self.activation = activation_function
		"""The activation function of the layer. Can be ReLU, softmax. """
		self.rng = self.__initialize_seed(seed)
		"""The random number generator. """
		self.W = np.zeros((output_dim, input_dim))
		"""The weight matrix. (output_dim, input_dim)"""
		self.b = np.zeros((output_dim, 1))
		"""The bias vector. (output_dim, 1)"""
		self.__initialize_model(input_dim=input_dim, output_dim=output_dim)
	

	def __initialize_seed(self, seed:int) -> np.random._generator.Generator:
		"""Initialize the random number generator. 
		
		Args:
			seed (int): The seed. 
		
		Returns:
			rng (Generator): The random number generator. """
		
		rng = np.random.default_rng()
		# get the BitGenerator used by default_rng
		BitGen = type(rng.bit_generator)
		# use the state from a fresh bit generator
		rng.bit_generator.state = BitGen(seed).state

		return rng


	def __initialize_model(self, input_dim:int, output_dim:int) -> None:
		"""Initialize the network. 
			
			Args:
				input_dim (int): Dimension of input. 
				output_dim (int): Dimension of output. """
		
		self.W = np.sqrt(2/(input_dim)) * self.rng.standard_normal(size = (output_dim, input_dim))
		self.b = np.zeros((output_dim, 1))


	def forward(self, X:np.ndarray) -> np.ndarray:
		"""Returns p and intermediary activation values. 
		
		Args:
			X (np.ndarray): The input data. (input_dim, n). 
					
		Returns:
			output (np.ndarray): The output of the perceptron layer. (output_dim, n)"""

		# Apply layer
		s = self.W @ X + self.b # (output_dim, input_dim) x (input_dim, n) = (output_dim, n)
		
		if self.activation == "ReLu":
			return np.maximum(s, 0)
		elif self.activation == "softmax":
			return np.exp(s) / np.sum(np.exp(s), axis=0)
		else:
			raise ValueError("No correct activation function given in the MLP. \
					The activation function given was: " + self.activation)
		

def test_mlp():
	"""Tests the MLP. """
	
	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)

	input_list = ['conv_flat', 'X1']
	output_list = ['X1', 'P']

	for i in range(len(input_list)):
		cur_input = load_data[input_list[i]]
		cur_output = load_data[output_list[i]]

		if output_list[i] == 'P':
			activation_function = "softmax"
			cur_input = cur_input.reshape((load_data['X1'].shape[1], load_data['X1'].shape[2]))
			W = load_data['W2']
			b = load_data['b2']
		else:
			activation_function = "ReLu"
			W = load_data['W1']
			b = load_data['b1']

		mlp = MLP_Layer(input_dim=cur_input.shape[0], output_dim=cur_output.shape[0], activation_function=activation_function)
		mlp.W = W
		mlp.b = b

		diff = cur_output - mlp.forward(X=cur_input)

		print("Diff with " + input_list[i] + " as input:", np.linalg.norm(diff))



if __name__ == "__main__":
	test_mlp()