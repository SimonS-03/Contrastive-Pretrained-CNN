import numpy as np
from dataloader import Dataloader


class Convolution_Layer:
	"""Create a convolution layer. Initilization done with He init.
	Args:
		filter_width: the number of units across for the filers.
		nf: number of filters
		activation_function: must be from {"ReLu"}"""
	def __init__(self, filter_width: int, nf: int, activation_function: str):
		self.filter_width = filter_width
		self.nf = nf
		self.filter_entries = filter_width**2 * 3 # Square filter, 3 channels
		
		rng = np.random.default_rng()
		seed = 42
		bit_gen = type(rng.bit_generator)
		rng.bit_generator.state = bit_gen(seed).state
		self.rng = rng

		self.F = rng.normal(loc=0, scale=np.sqrt(2/self.filter_entries), size=(self.filter_entries, self.nf)) 
		"""Filters. (filter_entries, nf)"""

		self.b = np.zeros(shape=(self.nf, 1))
		"""Filter bias. (filter_entries, nf)"""

		self.h = None
		"""Activation of conv. layer. (n_p*nf, batch_size)"""

		self.activation_function = activation_function
	
	def forward(self, Mx: np.ndarray) -> np.ndarray:
		n_patches = Mx.shape[0] # Mx has shape (n_patches, filter_entries, batch_size)
		pre_act = np.einsum("ijn, jl -> iln", Mx, self.F, optimize=True) + self.b.reshape(1, self.nf, 1) #(np, nf, batch_size)
		pre_act = pre_act.reshape((n_patches*self.nf, pre_act.shape[2]), order='C') # (n_p*n_f, batch_size)

		act = None
		if self.activation_function == "ReLu": 
			act = pre_act
			act[act < 0] *= 0
		else: # TODO: implement softmax?
			raise ValueError("Activation function in layer not registered/implemented, returning None.")

		self.h = act
		return act
		

def test_conv():
	"""Tests the MLP. """
	
	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)

	X = load_data['X'] # (d, n)
	Fs = load_data['Fs'] # (f, f, rgb, nf)

	f = Fs.shape[0]
	nf = 2

	loader = Dataloader()
	cur_input = loader.create_Mx(X=X, filter_width=f, image_width=32)# (d/f^2, f^2*rgb, n)

	cur_output = load_data['conv_flat']

	conv = Convolution_Layer(filter_width=f, nf=nf, activation_function="ReLu")
	conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	diff = cur_output - conv.forward(Mx=cur_input)

	print("Diff:", np.linalg.norm(diff))
			

if __name__ == "__main__":
	test_conv()