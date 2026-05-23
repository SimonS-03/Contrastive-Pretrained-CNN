import torch
from utils import dropout

class Conv_Layer:
	"""Create a convolution layer. Initilization done with He init.
	Args:
		filter_width: the number of units across for the filers.
		nf: number of filters
		activation_function: must be from {"ReLu"}"""
	def __init__(self, filter_width: int, nf: int, activation_function: str):
		self.filter_width = filter_width
		self.nf = nf
		self.filter_entries = filter_width**2 * 3 # Square filter, 3 channels
		self.n_p = None

		rng = torch.Generator()
		seed = 42
		rng.manual_seed(seed)
		self.rng = rng

		self.W = torch.normal(mean=0, std=torch.math.sqrt(2/self.filter_entries), size=(self.filter_entries, self.nf), generator=rng, requires_grad=True) 
		"""Filters. (filter_entries, nf)"""

		self.b = torch.zeros(size=(self.nf, 1), requires_grad=True)
		"""Filter bias. (filter_entries, nf)"""

		self.h = None
		"""Activation of conv. layer. (n_p*nf, batch_size)"""

		self.activation_function = activation_function
	
	def forward(self, Mx: torch.Tensor, dropout_prob: float) -> torch.Tensor:
		n_patches = Mx.shape[0] # Mx has shape (n_patches, filter_entries, batch_size)
		self.n_p = n_patches
		
		pre_act = torch.einsum("ijn, jl -> iln", Mx, self.W) + self.b.reshape(1, self.nf, 1) #(np, nf, batch_size)
		pre_act = pre_act.reshape(n_patches*self.nf, pre_act.shape[2]) # (n_p*n_f, batch_size)

		act = None
		if self.activation_function == "ReLu": 
			act = pre_act
			act[act < 0] *= 0
		else: # TODO: implement softmax?
			raise ValueError("Activation function in layer not registered/implemented, returning None.")

		self.h = dropout(act, dropout_prob)
		safety = 0
		while(torch.all(self.h == 0)):
			self.h = dropout(act, dropout_prob)
			safety += 1
			if safety > 100:
				print("Activation will be zero due to safety break in dropout loop!")
				break

		
		return self.h
			

if __name__ == "__main__":
	pass
