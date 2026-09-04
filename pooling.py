import torch

class Pool_Layer:
	"""Initialize the pooling layer. 
			
	Args:
		kernel (int): The kernel size (kernel x kernel). 
		stride (int): The stride. 
		padding (int): Padding of the image. 
		dropout_prob (float): Probability of a node being turned off for each update step.
		device (str): Not used in this layer, only added for consistency between layers. (cpu)"""


	def __init__(self, kernel:int, stride:int, padding:int, dropout_prob:float=0, device:str="cpu") -> None:

		self.nn_layer = torch.nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
		"""The built-in pooling layer. """

		self.dropout_prob = dropout_prob
		"""Probability of a node being turned off for each update step. """

		self.h:torch.Tensor = None
		"""Activation of the pooling layer. ([h-kernel]/stride+1, [w-kernel]/stride+1)"""

		self.device:str = device
		"""Not used in this layer, only added for consistency between layers."""


	def forward(self, X:torch.Tensor, apply_dropout:bool=True) -> torch.Tensor:
		"""Does the forward pass. 
		
		Args:
			X (torch.Tensor): The input "image". (n, nf, h, w)
			apply_dropout (bool): If dropout shall be applied. 
			
		Returns: 
			h (torch.Tensor): The output activations. [height-kernel+2\*padding]/stride+1, [width-kernel+2\*padding]/stride+1, number of filters, number of images"""

		self.h = self.nn_layer(X)

		self.h = torch.nn.functional.dropout(self.h, self.dropout_prob, training=apply_dropout)

		return self.h


if __name__ == "__main__":
	pass