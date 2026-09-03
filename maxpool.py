import torch
from utils import dropout

class Maxpool_Layer:
	"""Create a Max-Pool layer.
	Args:
		filter_width: the number of units across for the filers.
		nf: number of filters
		activation_function: must be from {"ReLu"}"""
	def __init__(self, filter_width: int, nf: int, channels: int, activation_function: str, input_width: int, stride: int = 1, padding: int = 1):
		self.filter_width = filter_width
		"""Width of the filter. """

		self.filter_entries = filter_width**2 # Square filter, 3 (or prev nf) channels
		"""Number of items in each filter. """
		self.n_p = None
		"""Number of filter patches per image. """
		self.input_width = input_width

		self.W = None
		"""Filters. (filter_entries, nf)"""

		self.b = None
		"""Filter bias. (filter_entries, nf)"""

		self.h = None
		"""Activation of conv. layer. (n_p*channels, batch_size)"""

		self.padding = padding 
		self.stride = stride
	
	def forward(self, Mx: torch.Tensor, dropout_prob: float) -> torch.Tensor:
		"""Does the forward pass.
		
		Args:
			Mx (torch.Tensor): The Mx matrix. (width*height/f^2, rgb*f^2, n)
			dropout_prob (float): Probability of a node being turned off for the forward pass.
		
		Returns:
			h (torch.Tensor): The output from the layer. (width*height/f^2*n_f, batch_size)"""
		
		n_patches = Mx.shape[0] # Mx has shape (n_patches, filter_entries, batch_size)
		self.n_p = n_patches
		
		pre_act = torch.einsum("ijn, jl -> iln", Mx, self.W) + self.b.reshape(1, self.nf, 1) #(np, nf, batch_size)
		pre_act_reshaped = pre_act.reshape(n_patches*self.nf, pre_act.shape[2]) # (n_p*n_f, batch_size)

		act = None
		if self.activation_function == "ReLu": 
			act = torch.relu(pre_act_reshaped)
		else: # TODO: implement softmax?
			raise ValueError("Activation function in layer not registered/implemented, returning None.")

		
		#act_drop = dropout(act, dropout_prob)
		#safety = 0
		#while(torch.all(act_drop == 0)):
		#	act_drop = dropout(act, dropout_prob)
		#	safety += 1
		#	if safety > 100:
		#		print("Activation will be zero due to safety break in dropout loop!")
		#		break

		self.h = torch.nn.functional.dropout(act, dropout_prob, training=True)
		return self.h
			

if __name__ == "__main__":
	pass
