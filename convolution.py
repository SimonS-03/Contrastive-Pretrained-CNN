import torch

class Conv_Layer:
	"""Create a convolution layer. Initilization done with He init.
	Args:
		filter_width (int): the number of units across for the filers.
		channels (int): Number of input channels. 
		nf (int): Number of filters, also number of output channels. 
		activation_function (str): must be from {"ReLu"}
		stride (int): Stride of the filter. 
		padding (int): Padding of the image. """
	
	def __init__(self, filter_width: int, channels:int, nf: int, activation_function: str, stride:int, padding:int):

		self.h:torch.Tensor = None
		"""Activation of conv. layer. (n_p*nf, batch_size)"""

		self.nn_layer = torch.nn.Conv2d(in_channels=channels, out_channels=nf, kernel_size=filter_width, stride=stride, padding=padding)
		"""The built-in convolution layer. """

		self.__initialize_layer()

		self.W:torch.Tensor = self.nn_layer.weight
		"""The weights of the network. """

		self.b:torch.Tensor = self.nn_layer.bias
		"""The bias of the network. """

		self.activation_function = activation_function


	def __initialize_layer(self) -> None:
		"""Call this function after initializing self.nn_layer. The function initializes the layer with He initialization. """

		# He/Kaiming initialization
		torch.nn.init.kaiming_normal_(self.nn_layer.weight, mode='fan_in', nonlinearity='relu')
		torch.nn.init.zeros_(self.nn_layer.bias)


	def forward(self, X:torch.Tensor, dropout_prob:float) -> torch.Tensor:
		"""Does the forward pass.
				
		Args:
			X (torch.Tensor): The input matrix. (height, width, number of filters, number of images)
			dropout_prob (float): Probability of an output node being turned off for the forward pass.
		
		Returns:
			h (torch.Tensor): The output from the layer. ([height-filter_width+2\*padding]/stride+1, [width-filter_width+2\*padding]/stride+1, number of filters, number of images)"""

		#Permute for the built in functions sake
		X_perm = X.permute(3, 2, 0, 1).contiguous() # (n, nf, h, w)

		#Do the convolution
		pre_act_perm:torch.Tensor = self.nn_layer(X_perm)

		#Permute back
		pre_act = pre_act_perm.permute(2, 3, 1, 0).contiguous() # (h, w, nf, n)

		act = None
		if self.activation_function == "ReLu": 
			act = torch.relu(pre_act)
		else: # TODO: implement softmax?
			raise ValueError("Activation function in layer not registered/implemented, returning None.")

		self.h = torch.nn.functional.dropout(act, dropout_prob, training=True)

		return self.h
			

if __name__ == "__main__":
	pass
