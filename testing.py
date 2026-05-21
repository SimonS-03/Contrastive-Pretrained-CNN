# A file for testing the other files/functions/classes

import matplotlib.pyplot as plt
import numpy as np
import pickle
import mlp, cnn, convolution, dataloader


def test_cyclic_lr() -> None:
	"""Test the increasing learning rate. """
	lr_list = []
	cnn_test = cnn.CNN([])
	for i in range(100+1):
		lr_list.append(cnn_test.get_cyclic_lr(min_lr=0.5, max_lr=1.5, cycle_length=10, iteration=i))

	plt.plot(lr_list)
	plt.title("Increasing learning rate")
	plt.ylabel("Learning rate")
	plt.xlabel("Iteration")
	plt.show()


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


def test_total_forward():
	"""The the forward pass with one convolution and two MLP layers. """

	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)
	X = load_data['X'] # (d, n)
	Fs = load_data['Fs'] # (f, f, rgb, nf)

	f = Fs.shape[0]
	nf = 2

	loader = dataloader.Dataloader()
	Mx = loader.create_Mx(X=X, filter_width=f, image_width=32)# (d/f^2, f^2*rgb, n)

	conv = convolution.Convolution_Layer(filter_width=f, nf=nf, activation_function="ReLu")
	conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	mlp_1 = mlp.MLP_Layer(input_dim=load_data['conv_flat'].shape[0], output_dim=load_data['X1'].shape[0], activation_function="ReLu")
	mlp_1.W = load_data['W1']
	mlp_1.b = load_data['b1']

	mlp_2 = mlp.MLP_Layer(input_dim=load_data['X1'].shape[0], output_dim=load_data['P'].shape[0], activation_function="softmax")
	mlp_2.W = load_data['W2']
	mlp_2.b = load_data['b2']

	cnn_test = cnn.CNN([conv, mlp_1, mlp_2])

	cnn_test.forward(Mx)

	print("conv diff:", np.linalg.norm(conv.h - load_data['conv_flat']))
	print("x1 diff:", np.linalg.norm(mlp_1.h - load_data['X1']))
	print("p diff:", np.linalg.norm(mlp_2.h - load_data['P']))


def test_total_backward():
	"""Test the total backward pass. """

	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)
	X = load_data['X'] # (d, n)
	Fs = load_data['Fs'] # (f, f, rgb, nf)

	f = Fs.shape[0]
	nf = 2

	loader = dataloader.Dataloader()
	Mx = loader.create_Mx(X=X, filter_width=f, image_width=32)# (d/f^2, f^2*rgb, n)

	conv = convolution.Conv_Layer(filter_width=f, nf=nf, activation_function="ReLu")
	conv.W = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	mlp_1 = mlp.MLP_Layer(input_dim=load_data['conv_flat'].shape[0], output_dim=load_data['X1'].shape[0], activation_function="ReLu")
	mlp_1.W = load_data['W1']
	mlp_1.b = load_data['b1']

	mlp_2 = mlp.MLP_Layer(input_dim=load_data['X1'].shape[0], output_dim=load_data['P'].shape[0], activation_function="softmax")
	mlp_2.W = load_data['W2']
	mlp_2.b = load_data['b2']

	cnn_test = cnn.CNN([conv, mlp_1, mlp_2])

	P = cnn_test.forward(Mx)

	test_grads = cnn_test.get_gradient(Mx_batch=Mx, Y_batch=load_data['Y'], P_batch=P)

	print("F grad diff:", np.linalg.norm(test_grads[0]['F'] - load_data['grad_Fs_flat']))
	print("Fb grad:", test_grads[0]['b'])


def test_overfit_small_sample():

	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)
	X = load_data['X'] # (d, n)
	Fs = load_data['Fs'] # (f, f, rgb, nf)

	f = Fs.shape[0]
	nf = 2

	n = 5

	loader = dataloader.Dataloader()
	Mx = loader.create_Mx(X=X, filter_width=f, image_width=32)[:, :, :n]# (d/f^2, f^2*rgb, n)

	conv = convolution.Conv_Layer(filter_width=f, nf=nf, activation_function="ReLu")
	#conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	mlp_1 = mlp.MLP_Layer(input_dim=load_data['conv_flat'].shape[0], output_dim=load_data['X1'].shape[1], activation_function="ReLu")

	mlp_2 = mlp.MLP_Layer(input_dim=load_data['X1'].shape[1], output_dim=load_data['P'].shape[0], activation_function="softmax")

	cnn_test = cnn.CNN([conv, mlp_1, mlp_2])

	cnn_test.minibatchGD(X_train=Mx, Y_train=load_data['Y'][:, :n], y_train=load_data['y'][:n], num_epochs=1000, X_val=Mx, \
					  Y_val=load_data['Y'][:, :n], y_val=load_data['y'][:n], batch_size=n, max_lr=0.001, min_lr=0.001)
	

def test_mirroring():
	"""Test mirroring data. """

	# Load a batch of training data
	cifar_dir = 'Datasets/cifar-10-batches-py/'
	with open(cifar_dir + 'data_batch_1', 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')

	# Extract the image data and cast to float from the dict dictionary
	X = dict[b'data'].astype(np.float64) / 255.0
	X = X.transpose()
	nn = X.shape[1]

	# Reshape each image from a column vector to a 3d array
	X_im = X.reshape((32, 32, 3, nn), order='F')
	X_im = np.transpose(X_im, (1, 0, 2, 3))

	X_mirror = dataloader.Dataloader().get_mirrored_data(X_im)

	# Display the first 5 images
	ni = 5
	fig, axs = plt.subplots(1, 5, figsize=(10, 5))
	for i in range(ni):
		axs[i].imshow(X_im[:, :, :, i])
	axs[i].axis('off')
	plt.show()

	# Display the first 5 images
	ni = 5
	fig, axs = plt.subplots(1, 5, figsize=(10, 5))
	for i in range(ni):
		axs[i].imshow(X_mirror[:, :, :, i])
	axs[i].axis('off')
	plt.show()
	

def test_rotation():
	"""Test mirroring data. """

	# Load a batch of training data
	cifar_dir = 'Datasets/cifar-10-batches-py/'
	with open(cifar_dir + 'data_batch_1', 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')

	# Extract the image data and cast to float from the dict dictionary
	X = dict[b'data'].astype(np.float64) / 255.0
	X = X.transpose()
	nn = X.shape[1]

	# Reshape each image from a column vector to a 3d array
	X_im = X.reshape((32, 32, 3, nn), order='F')
	X_im = np.transpose(X_im, (1, 0, 2, 3))

	X_rot = dataloader.Dataloader().get_rotated_data(X_im)

	x_jump = X_rot.shape[-1]//4

	for j in range(3+1):
		# Display the first 5 images
		ni = 5
		fig, axs = plt.subplots(1, 5, figsize=(10, 5))
		for i in range(ni):
			axs[i].imshow(X_rot[:, :, :, j*x_jump+i])
		axs[i].axis('off')
		plt.show()


def test_augmentation():
	"""Test mirroring data. """

	# Load a batch of training data
	cifar_dir = 'Datasets/cifar-10-batches-py/'
	with open(cifar_dir + 'data_batch_1', 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')

	# Extract the image data and cast to float from the dict dictionary
	X = dict[b'data'].astype(np.float64) / 255.0
	X = X.transpose()
	nn = X.shape[1]

	# Reshape each image from a column vector to a 3d array
	X_im = X.reshape((32, 32, 3, nn), order='F')
	X_im = np.transpose(X_im, (1, 0, 2, 3))

	X_aug = dataloader.Dataloader().augment_data(X_im)

	x_jump = X_aug.shape[-1]//8

	for j in range(7+1):
		# Display the first 5 images
		ni = 5
		fig, axs = plt.subplots(1, 5, figsize=(10, 5))
		for i in range(ni):
			axs[i].imshow(X_aug[:, :, :, j*x_jump+i])
		axs[i].axis('off')
		plt.show()



if __name__ == "__main__":
	test_augmentation()
