# A file for testing the other files/functions/classes

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import mlp, cnn, convolution, dataloader
import utils

def test_cyclic_lr() -> None:
	"""Test the increasing learning rate. """
	
	utils.cyclic_lr(cycle_length=10, min_lr=0.5, max_lr=1.5).plot_lr(100)


def test_increasing_lr() -> None:
	"""Test the increasing learning rate. """
	
	utils.cyclic_increasing_lr(starting_cycle_length=10, min_lr=0.5, max_lr=1.5).plot_lr(300)


def test_conv():
	"""Tests the MLP. """
	
	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)

	X = torch.tensor(load_data['X']).float() # (d, n)
	Fs = torch.tensor(load_data['Fs']).float() # (f, f, rgb, nf)

	f = Fs.shape[0]
	nf = 2

	loader = dataloader.Dataloader()
	X_ims = X.reshape(3, 32, 32, X.shape[-1]).permute(2, 1, 0, 3)
	cur_input = loader.create_Mx(X=X_ims, filter_width=f)# (d/f^2, f^2*rgb, n)

	cur_output = torch.tensor(load_data['conv_flat']).float()

	conv = convolution.Conv_Layer(filter_width=f, nf=nf, activation_function="ReLu")
	conv.W = Fs.reshape(f*f*3, nf) # (f^2*rgb, nf)

	diff = cur_output - conv.forward(Mx=cur_input)

	print("Diff:", torch.linalg.norm(diff).item())


def test_mlp():
	"""Tests the MLP. """
	
	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)

	input_list = ['conv_flat', 'X1']
	output_list = ['X1', 'P']

	for i in range(len(input_list)):
		cur_input = torch.tensor(load_data[input_list[i]]).float()
		cur_output = torch.tensor(load_data[output_list[i]]).float()

		if output_list[i] == 'P':
			activation_function = "softmax"
			cur_input = cur_input.reshape((load_data['X1'].shape[1], load_data['X1'].shape[2]))
			W = torch.tensor(load_data['W2']).float()
			b = torch.tensor(load_data['b2']).float()
		else:
			activation_function = "ReLu"
			W = torch.tensor(load_data['W1']).float()
			b = torch.tensor(load_data['b1']).float()

		mlp_test = mlp.MLP_Layer(input_dim=cur_input.shape[0], output_dim=cur_output.shape[0], activation_function=activation_function)
		mlp_test.W = W
		mlp_test.b = b

		diff = cur_output - mlp_test.forward(X=cur_input)

		print("Diff with " + input_list[i] + " as input:", torch.linalg.norm(diff).item())


def test_total_forward():
	"""The the forward pass with one convolution and two MLP layers. """

	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)
	X = torch.tensor(load_data['X']).float() # (d, n)
	Fs = torch.tensor(load_data['Fs']).float() # (f, f, rgb, nf)

	f = Fs.shape[0]
	nf = 2

	loader = dataloader.Dataloader()
	X = X.T
	X_ims = X.reshape(X.shape[0], 3, 32, 32).permute(2, 3, 1, 0)
	Mx = loader.create_Mx(X=X_ims, filter_width=f)# (d/f^2, f^2*rgb, n)

	conv = convolution.Conv_Layer(filter_width=f, nf=nf, activation_function="ReLu")
	conv.W = Fs.reshape(f*f*3, nf) # (f^2*rgb, nf)

	mlp_1 = mlp.MLP_Layer(input_dim=load_data['conv_flat'].shape[0], output_dim=load_data['X1'].shape[0], activation_function="ReLu")
	mlp_1.W = torch.tensor(load_data['W1']).float()
	mlp_1.b = torch.tensor(load_data['b1']).float()

	mlp_2 = mlp.MLP_Layer(input_dim=load_data['X1'].shape[0], output_dim=load_data['P'].shape[0], activation_function="softmax")
	mlp_2.W = torch.tensor(load_data['W2']).float()
	mlp_2.b = torch.tensor(load_data['b2']).float()

	cnn_test = cnn.CNN([conv, mlp_1, mlp_2])

	cnn_test.forward(Mx, dropout_prob=0)

	print("conv diff:", torch.linalg.norm(conv.h - torch.tensor(load_data['conv_flat']).float()).item())
	print("x1 diff:", torch.linalg.norm(mlp_1.h - torch.tensor(load_data['X1']).float()).item())
	print("p diff:", torch.linalg.norm(mlp_2.h - torch.tensor(load_data['P']).float()).item())


def test_total_backward():
	"""Test the total backward pass. """

	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)
	X = torch.tensor(load_data['X']).float() # (d, n)
	Fs = torch.tensor(load_data['Fs']).float() # (f, f, rgb, nf)

	f = Fs.shape[0]
	nf = 2

	loader = dataloader.Dataloader()
	X = X.T
	X_ims = X.reshape(X.shape[0], 3, 32, 32).permute(2, 3, 1, 0)
	Mx = loader.create_Mx(X=X_ims, filter_width=f)# (d/f^2, f^2*rgb, n)

	conv = convolution.Conv_Layer(filter_width=f, nf=nf, activation_function="ReLu")
	conv.W = Fs.reshape(f*f*3, nf) # (f^2*rgb, nf)

	mlp_1 = mlp.MLP_Layer(input_dim=load_data['conv_flat'].shape[0], output_dim=load_data['X1'].shape[0], activation_function="ReLu")
	mlp_1.W = torch.tensor(load_data['W1']).float()
	mlp_1.b = torch.tensor(load_data['b1']).float()

	mlp_2 = mlp.MLP_Layer(input_dim=load_data['X1'].shape[0], output_dim=load_data['P'].shape[0], activation_function="softmax")
	mlp_2.W = torch.tensor(load_data['W2']).float()
	mlp_2.b = torch.tensor(load_data['b2']).float()

	cnn_test = cnn.CNN([conv, mlp_1, mlp_2])

	P = cnn_test.forward(Mx, dropout_prob=0)

	test_grads = cnn_test.get_gradient(Mx_batch=Mx, Y_batch=torch.tensor(load_data['Y']).float(), P_batch=P)

	print("F grad diff:", torch.linalg.norm(test_grads[0]['W'] - torch.tensor(load_data['grad_Fs_flat']).float()).item())
	print("Fb grad:", test_grads[0]['b'])


def test_overfit_small_sample():
	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)
	X = torch.tensor(load_data['X']).float() # (d, n)
	Fs = torch.tensor(load_data['Fs']).float() # (f, f, rgb, nf)

	f = Fs.shape[0]
	nf = 2
	n = 100

	loader = dataloader.Dataloader()
	X = X.T
	X_ims = X.reshape(X.shape[0], 3, 32, 32).permute(2, 3, 1, 0)

	conv = convolution.Conv_Layer(filter_width=f, nf=nf, activation_function="ReLu")
	#conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	mlp_1 = mlp.MLP_Layer(input_dim=load_data['conv_flat'].shape[0], output_dim=load_data['X1'].shape[1], activation_function="ReLu")

	mlp_2 = mlp.MLP_Layer(input_dim=load_data['X1'].shape[1], output_dim=load_data['P'].shape[0], activation_function="softmax")

	cnn_test = cnn.CNN([conv, mlp_1, mlp_2])
	
	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=5, min_lr=5e-3, max_lr=5e-2)
	adamW_optim = utils.AdamW()
	cnn_test.minibatchGD(X_train=X_ims, Y_train=torch.tensor(load_data['Y']).float()[:, :n], y_train=torch.tensor(load_data['y']).float()[:n], X_val=X_ims, \
					  Y_val=torch.tensor(load_data['Y']).float()[:, :n], y_val=torch.tensor(load_data['y']).float()[:n], schedule=cyclic_schedule, optim=adamW_optim, batch_size=5, max_num_epochs=200)

def test_mirroring():
	"""Test mirroring data. """

	# Load a batch of training data
	cifar_dir = 'Datasets/cifar-10-batches-py/'
	with open(cifar_dir + 'data_batch_1', 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')

	# Extract the image data and cast to float from the dict dictionary
	X = torch.tensor(dict[b'data']).float() / 255.0 # (n, rgb*height*width)

	# Reshape each image from a column vector to a 3d array
	X_im = X.reshape(X.shape[0], 3, 32, 32).permute(2, 3, 1, 0) # (n, rgb, height, width) -> (height, width, rgb, n)

	X_mirror = X_im.clone()
	dataloader.Dataloader().mirror_data(X_mirror, mirror_prob=1)

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
	X = dict[b'data'].astype(torch.float64) / 255.0
	X = X.transpose()
	nn = X.shape[1]

	# Reshape each image from a column vector to a 3d array
	X_im = X.reshape((32, 32, 3, nn), order='F')
	X_im = torch.transpose(X_im, (1, 0, 2, 3))

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
	X = torch.tensor(dict[b'data']).float() / 255.0
	X = X.T
	nn = X.shape[1]

	# Reshape each image from a column vector to a 3d array
	X_im = X.T.reshape(nn, 3, 32, 32).permute(3, 2, 1, 0)
	X_im = X_im.permute(1, 0, 2, 3)

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


def test_tiny_image():
	"""Load some of the tiny images. """
	dataload = dataloader.Dataloader()
	data = dataload.load_train_data()
	dataload.show_tiny_images(data['X_train'])


def test_augment_tiny():
	"""Augment tiny images. """
	dataload = dataloader.Dataloader()
	data = dataload.load_train_data(print_progress=True)
	X_mini = data['X_train'][..., :5]
	X_mini = dataload.normalize(X_mini)
	aug_X = dataload.augment_data(X_mini, flip_prob=0, rot_prob=0, noise_prob=1)
	dataload.show_tiny_images(X_mini)
	dataload.show_tiny_images(aug_X)


def test_train_augment_tiny():
	"""Augment tiny images. """
	dataload = dataloader.Dataloader()
	data = dataload.load_train_data(print_progress=True)
	X_mini = data['X_train'][..., :5]
	X_mini = dataload.normalize(X_mini)
	aug_X = dataload.augment_data(X_mini)
	dataload.show_tiny_images(X_mini)
	dataload.show_tiny_images(aug_X)


def test_Mx():
	"""Test the new "get Mx". """


	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)
	X = torch.tensor(load_data['X']).float() # (d, n)
	Fs = torch.tensor(load_data['Fs']).float() # (f, f, rgb, nf)

	f = Fs.shape[0]

	X = X.T
	X_ims = X.reshape(X.shape[0], 3, 32, 32).permute(2, 3, 1, 0)
	ni = X_ims.shape[3]
	Mx = torch.zeros(size=((32//f)**2, f*f*3, ni))
	for img_index in range(ni):
		img = X_ims[:, :, :, img_index]
		l = 0
		for i in range(32//f):
			for j in range(32//f):
				patch = img[i*f:i*f+f, j*f:j*f+f, :]
				Mx[l, :, img_index] = patch.reshape(1, f*f*3)
				l += 1

	loader = dataloader.Dataloader()

	Mx_new = loader.create_Mx(X_ims, f)

	print("Diff:", torch.linalg.norm(Mx - Mx_new).item())


def test_contrastive_small_sample():
	
	f = 4
	nf = 160
	hid_dim = 100
	loader = dataloader.Dataloader()
	data = loader.load_train_data(print_progress=True)
	X_train = data["X_train"][..., :40000]
	X_val = data["X_val"][..., :1000]
	
	X_train = loader.normalize(X_train)
	X_val = loader.normalize(X_val, use_train_data=True)
	
	conv1 = convolution.Conv_Layer(filter_width=4, nf=nf, channels=3, activation_function="ReLu")
	conv2 = convolution.Conv_Layer(filter_width=4, nf=nf, channels=nf, activation_function="ReLu")
	loader.show_filters(conv1.W)
	n_p = (X_train.shape[0]//(4*4))**2 # NEEDS TO BE UPDATED WHEN CONV LAYERS CHANGE
	
	mlp_1 = mlp.MLP_Layer(input_dim=n_p*nf, output_dim=hid_dim, activation_function="ReLu")
	mlp_2 = mlp.MLP_Layer(input_dim=hid_dim, output_dim=200, activation_function="softmax")

	cnn_test = cnn.CNN([conv1, conv2, mlp_1, mlp_2], lam=1e-3, dropout_prob=.3)
	cnn_test.dataloader = loader
	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=25, min_lr=1e-4, max_lr=1e-1)
	adamW_optim = utils.AdamW()
	
	cnn_test.contrastiveLearning(X_train=X_train, \
					schedule=cyclic_schedule, optim=adamW_optim, batch_size=256, max_num_epochs=10)
	cnn_test.dataloader.show_filters(cnn_test.layers[0].W, freeze=True)

if __name__ == "__main__":
	test_augment_tiny()
