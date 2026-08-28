import torch

class Pool_Layer:
	"""Initialize the pooling layer. 
			
	Args:
		kernel (int): The kernel size (kernel x kernel). 
		stride (int): The stride. 
		padding (int): Padding of the image. 
		devide (str): Not used in this layer, only added for consistency between layers. (cpu)"""


	def __init__(self, kernel:int, stride:int, padding:int, device:str="cpu") -> None:

		self.nn_layer = torch.nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
		"""The built-in pooling layer. """

		self.h:torch.Tensor = None
		"""Activation of the pooling layer. ([h-kernel]/stride+1, [w-kernel]/stride+1)"""

		self.device:str = device
		"""Not used in this layer, only added for consistency between layers."""


	def forward(self, X:torch.Tensor, dropout_prob:float=0) -> torch.Tensor:
		"""Does the forward pass. 
		
		Args:
			X (torch.Tensor): The input "image". (n, nf, h, w)
			dropout_prob (float): Is not used in pooling, but needed for coding consistency between layers. 
			
		Returns: 
			h (torch.Tensor): The output activations. [height-kernel+2\*padding]/stride+1, [width-kernel+2\*padding]/stride+1, number of filters, number of images"""

		self.h = self.nn_layer(X)

		return self.h


if __name__ == "__main__":
	pass