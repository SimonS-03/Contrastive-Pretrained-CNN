import torch
from utils import dropout
class MLP_Layer:
	"""One layer of an MLP. 
	
	Args:
		input_dim (int): Dimension of input. 
		output_dim (int): Dimension of output. 
		activation_function (str): The activation function of the layer. Can be ReLu, softmax.
		device (str): Which device to run the layer on. (cpu)"""


	def __init__(self, input_dim:int, output_dim:int, activation_function:str, device:str="cpu"):
		"""Initialize the MLP layer.  """
		
		self.activation_function = activation_function
		"""The activation function of the layer. Can be ReLu, softmax. """
		self.device:str = device
		"""The device that the layer runs on. """
		self.W = None
		"""The weight matrix. (output_dim, input_dim)"""
		self.b = None
		"""The bias vector. (output_dim, 1)"""
		self.h = None
		"""The activation output. (output_dim, n)"""
		self.__initialize_model(input_dim=input_dim, output_dim=output_dim)


	def __initialize_model(self, input_dim:int, output_dim:int) -> None:
		"""Initialize the network. 
			
			Args:
				input_dim (int): Dimension of input. 
				output_dim (int): Dimension of output. """
		if self.activation_function in ["softmax", "linear"]:
			self.W = 0.01 * torch.randn(size=(output_dim, input_dim), device=self.device)
		else:
			self.W = torch.math.sqrt(2/(input_dim)) * torch.randn(size = (output_dim, input_dim), device=self.device)
		
		self.W.requires_grad = True
		self.b = torch.zeros((output_dim, 1), device=self.device, requires_grad=True)


	def forward(self, X:torch.Tensor, dropout_prob: float=0, check_nan: bool=False) -> torch.Tensor:
		"""Returns p and intermediary activation values. 
		
		Args:
			X (torch.Tensor): The input data. (input_dim, n). 
			dropout_prob (float): Probability of an output node being turned off for the forward pass.
			check_nan (bool): If we should check for NaN in the activation nodes. It is time inefficient. 
					
		Returns:
			output (torch.Tensor): The output of the perceptron layer. (output_dim, n)"""

		# Apply layer
		s = self.W @ X + self.b # (output_dim, input_dim) x (input_dim, n) = (output_dim, n)
		
		act = None
		if self.activation_function == "ReLu":
			act = torch.relu(s)
		elif self.activation_function == "softmax":
			act = torch.softmax(s, dim=0)
			self.h = act
			if check_nan and torch.any(self.h.isnan()):
				raise ValueError("Some values are NaN after doing softmax. ")
			return self.h
		elif self.activation_function == "linear":
			act = s
			self.h = act
			return self.h
		else:
			raise ValueError("No correct activation function given in the MLP. \
					The activation function given was: " + self.activation_function)

		if act is None:
			raise ValueError("activation is None for some reason if MLP forward!")
		

		#act_drop = dropout(act, dropout_prob)
		#safety = 0
		#while(torch.all(act_drop == 0)): # We dont want the entire activation to be zero!
		#	act_drop = dropout(act, dropout_prob)
		#	safety += 1
		#	if safety > 100:
		#		print("Activation will be zero due to safety break in dropout loop!")
		#		break
		
		self.h = torch.nn.functional.dropout(act, p=dropout_prob, training=True)

		if check_nan and torch.any(self.h.isnan()):
			raise ValueError("Some values are NaN after normal MLP. ")

		return self.h



if __name__ == "__main__":
	pass
