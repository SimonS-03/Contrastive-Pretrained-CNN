import numpy as np

class MLP_Layer:
	"""On layer of an MLP. """
	def __init__(self, n_inputs:int, n_outputs:int, activation_function:str, seed:int=42):
		"""Initialize the MLP layer. 
		
		Args:
			n_inputs (int): Number of inputs. 
			n_outputs (int): Number of outputs. 
			activation_function (str): The activation function of the layer. Can be ReLU, softmax.
			seed (int): The seed for the random number generator. """
		
		self.activation = activation_function
		"""The activation function of the layer. Can be ReLU, softmax. """
		self.rng = self.__initialize_seed(seed)
		"""The random number generator. """
		self.W = np.zeros((n_outputs, n_inputs))
		"""The weight matrix. (n_outputs, n_inputs)"""
		self.b = np.zeros((n_outputs, 1))
		"""The bias vector. (n_outputs, 1)"""
		self.__initialize_model(n_inputs=n_inputs, n_outputs=n_outputs)
	

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


	def __initialize_model(self, n_inputs:int, n_outputs:int) -> None:
		"""Initialize the network. 
			
			Args:
				n_inputs (int): Number of inputs. 
			n_outputs (int): Number of outputs. """
		
		self.W = np.sqrt(2/(n_inputs)) * self.rng.standard_normal(size = (n_outputs, n_inputs))
		self.b = np.zeros((n_outputs, 1))


	def forward(self, X:np.ndarray) -> np.ndarray:
		"""Returns p and intermediary activation values. 
		
		Args:
			X (np.ndarray): The input data. (n_inputs, n). 
					
		Returns:
			output (np.ndarray): The output of the perceptron layer. (n_outputs, n)"""

		# Apply layer
		s = self.W @ X + self.b # (n_outputs, n_inputs) x (n_inputs, n) = (n_outputs, n)
		
		if self.activation == "ReLu":
			return np.maximum(s, 0)
		elif self.activation == "softmax":
			return np.exp(s) / np.sum(np.exp(s), axis=0)
		else:
			raise ValueError("No correct activation function given in the MLP. \
					The activation function given was: " + self.activation)
		

if __name__ == "__main__":
	mlp = MLP_Layer(activation_function="Ys")
	mlp.__initialize_seed(42)