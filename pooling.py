import torch

class Pool_Layer:
	"""Initialize the pooling layer. 
			
	Args:
		kernel (int): The kernel size (kernel x kernel). 
		stride (int): The stride. """


	def __init__(self, kernel:int, stride:int) -> None:
		self.nn_layer = torch.nn.MaxPool2d(kernel_size=kernel, stride=stride)
		"""The built-in pooling layer. """

		self.h:torch.Tensor = None
		"""Activation of the pooling layer. ([h-kernel]/stride+1, [w-kernel]/stride+1)"""

		self.W:torch.Tensor = self.nn_layer.weight
		"""The weights of the network. """

		self.b:torch.Tensor = self.nn_layer.bias
		"""The bias of the network. """


	def forward(self, X:torch.Tensor, dropout_prob:float=0) -> torch.Tensor:
		"""Does the forward pass. 
		
		Args:
			X (torch.Tensor): The input "image". (h, w, nf, n)
			dropout_prob (float): Is not used in pooling, but needed for coding consistency between layers. """

		#Permute for the built in functions sake
		X_perm = X.permute(3, 2, 0, 1).contiguous() # (n, nf, h, w)

		act_perm:torch.Tensor = self.nn_layer(X_perm)

		#Permute back
		self.h = act_perm.permute(2, 3, 1, 0).contiguous() # (h, w, nf, n)

		return self.h


if __name__ == "__main__":
	pass