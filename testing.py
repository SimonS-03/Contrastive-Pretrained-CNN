# A file for testing the other files/functions/classes

import matplotlib.pyplot as plt
import numpy as np
import mlp, cnn, convolution, dataloader


def test_increasing_lr() -> None:
	"""Test the increasing learning rate. """
	lr_list = []
	cnn_test = cnn.CNN([])
	for i in range(10000):
		lr_list.append(cnn_test.get_increasing_cyclic_lr(min_lr=0, max_lr=1, starting_cycle_length=10, iteration=i))

	plt.plot(lr_list)
	plt.title("Increasing learning rate")
	plt.ylabel("Learning rate")
	plt.xlabel("Iteration")
	plt.show()


def test_conv():
	"""Tests the MLP. """
	
	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)

	X = load_data['X'] # (d, n)
	Fs = load_data['Fs'] # (f, f, rgb, nf)

	f = Fs.shape[0]
	nf = 2

	loader = dataloader.Dataloader()
	cur_input = loader.create_Mx(X=X, filter_width=f, image_width=32)# (d/f^2, f^2*rgb, n)

	cur_output = load_data['conv_flat']

	conv = convolution.Convolution_Layer(filter_width=f, nf=nf, activation_function="ReLu")
	conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	diff = cur_output - conv.forward(Mx=cur_input)

	print("Diff:", np.linalg.norm(diff))


def test_mlp():
	"""Tests the MLP. """
	
	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)

	input_list = ['conv_flat', 'X1']
	output_list = ['X1', 'P']

	for i in range(len(input_list)):
		cur_input = load_data[input_list[i]]
		cur_output = load_data[output_list[i]]

		if output_list[i] == 'P':
			activation_function = "softmax"
			cur_input = cur_input.reshape((load_data['X1'].shape[1], load_data['X1'].shape[2]))
			W = load_data['W2']
			b = load_data['b2']
		else:
			activation_function = "ReLu"
			W = load_data['W1']
			b = load_data['b1']

		mlp_test = mlp.MLP_Layer(input_dim=cur_input.shape[0], output_dim=cur_output.shape[0], activation_function=activation_function)
		mlp_test.W = W
		mlp_test.b = b

		diff = cur_output - mlp_test.forward(X=cur_input)

		print("Diff with " + input_list[i] + " as input:", np.linalg.norm(diff))


if __name__ == "__main__":
	test_mlp()