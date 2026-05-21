import numpy as np
from utils import dropout
class MLP_Layer:
	"""One layer of an MLP. """


	def __init__(self, input_dim:int, output_dim:int, activation_function:str, seed:int=42):
		"""Initialize the MLP layer. 
		
		Args:
			input_dim (int): Dimension of input. 
			output_dim (int): Dimension of output. 
			activation_function (str): The activation function of the layer. Can be ReLU, softmax.
			seed (int): The seed for the random number generator. """
		
		self.activation_function = activation_function
		"""The activation function of the layer. Can be ReLU, softmax. """
		self.rng = self.__initialize_seed(seed)
		"""The random number generator. """
		self.W = None
		"""The weight matrix. (output_dim, input_dim)"""
		self.b = None
		"""The bias vector. (output_dim, 1)"""
		self.h = None
		"""The activation output. (output_dim, n)"""
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


	def forward(self, X:np.ndarray, dropout_prob: float) -> np.ndarray:
		"""Returns p and intermediary activation values. 
		
		Args:
			X (np.ndarray): The input data. (input_dim, n). 
					
		Returns:
			output (np.ndarray): The output of the perceptron layer. (output_dim, n)"""

		# Apply layer
		s = self.W @ X + self.b # (output_dim, input_dim) x (input_dim, n) = (output_dim, n)
		
		act = None
		if self.activation_function == "ReLu":
			act = np.maximum(s, 0)
		elif self.activation_function == "softmax":
			act = np.exp(s) / np.sum(np.exp(s), axis=0)
			self.h = act
			return self.h
		else:
			raise ValueError("No correct activation function given in the MLP. \
					The activation function given was: " + self.activation_function)

		if act is None:
			raise ValueError("activation is None for some reason if MLP forward!")
		
		self.h = dropout(act, dropout_prob)
		safety = 0
		while(np.all(self.h == 0)): # We dont want the entire activation to be zero!
			self.h = dropout(act, dropout_prob)
			safety += 1
			if safety > 100:
				print("Activation will be zero due to safety break in dropout loop!")
				break
		
		return self.h



if __name__ == "__main__":
	pass
