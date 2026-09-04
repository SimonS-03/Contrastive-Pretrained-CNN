# A file for testing the other files/functions/classes

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import mlp, cnn, convolution, dataloader, pooling, utils
import utils
import math

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
	torch.manual_seed(42)

	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)
	X = torch.tensor(load_data['X']).float() # (d, n)
	#Fs = torch.tensor(load_data['Fs']).float() # (f, f, rgb, nf)

	#f = Fs.shape[0]
	f = 3
	stride = 1
	padding = 1
	#nf = 2
	nf = 128
	n = 100
	rgb = 3
	device = "mps"

	loader = dataloader.Dataloader()
	X = X.T
	X_ims = X.reshape(X.shape[0], 3, 32, 32).permute(2, 3, 1, 0).contiguous().to(device)
	loader.normalize(X=X_ims, use_train_data=False)

	conv_1 = convolution.Conv_Layer(filter_width=f, channels=rgb, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	#conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	conv_2 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)

	pool_1 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	conv_3 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	conv_4 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)

	pool_2 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	conv_5 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	pool_3 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	#mlp_1 = mlp.MLP_Layer(input_dim=load_data['conv_flat'].shape[0], output_dim=load_data['X1'].shape[1], activation_function="ReLu")
	mlp_1 = mlp.MLP_Layer(input_dim=(32//2**3)**2*nf, output_dim=400, activation_function="ReLu")

	mlp_2 = mlp.MLP_Layer(input_dim=400, output_dim=400, activation_function="ReLu")

	mlp_3 = mlp.MLP_Layer(input_dim=400, output_dim=load_data['P'].shape[0], activation_function="softmax")

	cnn_test = cnn.CNN([conv_1, conv_2, pool_1, conv_3, conv_4, pool_2, conv_5, pool_3, mlp_1, mlp_2, mlp_3], device=device)

	cnn_test.dataloader = loader
	
	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=5, min_lr=1e-4, max_lr=1e-3)
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
	data = dataload.load_train_data(print_progress=True, train_lim=5)
	dataload.show_tiny_images(data['X_train'])
	


def test_augment_tiny():
	"""Augment tiny images. """
	dataload = dataloader.Dataloader()
	data = dataload.load_train_data(train_lim=10, print_progress=True)
	X_mini = data['X_train']
	X_mini = dataload.normalize(X_mini)
	aug_X = dataload.augment_data(X_mini, flip_prob=.5, rot_prob=.5, noise_prob=1)
	print(f"Non-augmented shape: {X_mini.shape}")
	print(f"Augmented shape: {aug_X.shape}")
	dataload.show_tiny_images(X_mini, n=10)
	dataload.show_tiny_images(aug_X, n=10)


def test_train_augment_tiny():
	"""Augment tiny images. """
	torch.manual_seed(42)

	device = "mps"

	dataload = dataloader.Dataloader()
	#data = dataload.load_train_data(print_progress=True, train_lim=10000, val_lim=2000)
	data = dataload.load_cifar_data()

	X_train = dataload.normalize(data['X_train'])
	del data['X_train']
	X_val = dataload.normalize(data['X_val'], use_train_data=True)
	del data['X_val']
	dataload.set_normal_device(device=device)

	batch_size = 200
	n_train = X_train.shape[-1]

	f = 3
	nf = 128
	stride = 1
	padding = 1
	rgb = 3
	lam = 2.5e-4
	n_classes = data['Y_train'].shape[0]
	width = X_train.shape[0]

	conv_1 = convolution.Conv_Layer(filter_width=f, channels=rgb, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	#conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	conv_2 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)

	pool_1 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	conv_3 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	conv_4 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)

	pool_2 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	conv_5 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	pool_3 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	#mlp_1 = mlp.MLP_Layer(input_dim=load_data['conv_flat'].shape[0], output_dim=load_data['X1'].shape[1], activation_function="ReLu")
	mlp_1 = mlp.MLP_Layer(input_dim=(width//2**3)**2*nf, output_dim=400, activation_function="ReLu")

	mlp_2 = mlp.MLP_Layer(input_dim=400, output_dim=400, activation_function="ReLu")

	mlp_3 = mlp.MLP_Layer(input_dim=400, output_dim=n_classes, activation_function="linear")

	cnn_test = cnn.CNN([conv_1, conv_2, pool_1, conv_3, conv_4, pool_2, conv_5, pool_3, mlp_1, mlp_2, mlp_3], \
					lam=lam, device=device, dropout_prob=0.5)
	
	cnn_test.dataloader = dataload

	k = 2
	iters_per_epoch = n_train / batch_size
	starting_cycle_length = k * iters_per_epoch    # k in [2, 10]
	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=starting_cycle_length, min_lr=1e-4, max_lr=1e-3)
	adamW_optim = utils.AdamW()
	cnn_test.minibatchGD(X_train=X_train, Y_train=data['Y_train'], y_train=data['y_train'], X_val=X_val, \
					  Y_val=data['Y_val'], y_val=data['y_val'], schedule=cyclic_schedule, optim=adamW_optim, \
						batch_size=batch_size, max_num_epochs=200, show_plots=True, profile=False, debug_logs=False)

def test_train_contrastive_tiny(checkpoint: str = None):
	"""Augment tiny images. Checkpoint should be filepath to load/save """
	torch.manual_seed(42)

	device = "mps"

	dataload = dataloader.Dataloader()
	data = dataload.load_train_data(print_progress=True, train_lim=200, val_lim=100)
	X_train = dataload.normalize(data['X_train'])
	del data['X_train']
	X_val = dataload.normalize(data['X_val'], use_train_data=True)
	del data['X_val']
	dataload.set_normal_device(device=device)

	f = 3
	nf = 128
	stride = 1
	padding = 1
	rgb = 3
	lam = 0
	n_classes = data['Y_train'].shape[0]
	width = X_train.shape[0]

	conv_1 = convolution.Conv_Layer(filter_width=f, channels=rgb, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	#conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	conv_2 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)

	pool_1 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	conv_3 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	conv_4 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)

	pool_2 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	conv_5 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	pool_3 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	#mlp_1 = mlp.MLP_Layer(input_dim=load_data['conv_flat'].shape[0], output_dim=load_data['X1'].shape[1], activation_function="ReLu")
	mlp_1 = mlp.MLP_Layer(input_dim=(width//2**3)**2*nf, output_dim=400, activation_function="ReLu")

	mlp_2 = mlp.MLP_Layer(input_dim=400, output_dim=400, activation_function="ReLu")

	mlp_3 = mlp.MLP_Layer(input_dim=400, output_dim=n_classes, activation_function="softmax")


	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=5, min_lr=1e-2, max_lr=1e-1)
	adamW_optim = utils.AdamW()

	cnn_test = cnn.CNN([conv_1, conv_2, pool_1, conv_3, conv_4, pool_2, conv_5, pool_3, mlp_1, mlp_2, mlp_3], \
					lam=lam, device=device, lr_schedule=cyclic_schedule, adamW_optim=adamW_optim)
	
	cnn_test.dataloader = dataload
	

	if checkpoint: 
		try:
			cnn_test.load_checkpoint(checkpoint)
		except:
			print(f"Could not find checkpoint with filename {checkpoint}. Starting from scratch.")

	try: 
		cnn_test.contrastiveLearning(X_train=X_train, X_val=X_val, \
					batch_size=100, max_num_epochs=20, show_plots=True, debug_logs=False)
		cnn_test.show_filters(cnn_test.layers[0].W, freeze=True)
		if checkpoint: cnn_test.save_checkpoint(checkpoint)
	except KeyboardInterrupt:
		print("Training interupted...")
		if checkpoint: cnn_test.save_checkpoint(checkpoint)

def real_minibatch_run():
	"""Does a solid minibatch run"""

	dataload = dataloader.Dataloader()
	data = dataload.load_train_data(print_progress=True, train_lim=1000, val_lim=100)
	X_train = data['X_train']
	X_train = dataload.normalize(X_train)
	X_val = dataload.normalize(data['X_val'], use_train_data=True)

	f = 4
	nf = 100
	m = 100
	rgb = 3
	n_classes = data['Y_train'].shape[0]
	width = X_train.shape[0]

	conv = convolution.Conv_Layer(filter_width=f, nf=nf, channels=rgb, activation_function="ReLu")
	#conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	mlp_1 = mlp.MLP_Layer(input_dim=(width//f)**2*nf, output_dim=m, activation_function="ReLu")

	mlp_2 = mlp.MLP_Layer(input_dim=m, output_dim=n_classes, activation_function="softmax")

	cnn_final = cnn.CNN([conv, mlp_1, mlp_2], lam=0.01, dropout_prob=0.4)
	cnn_final.load_checkpoint("test_second_minibatch")
	cnn_final.dataloader = dataload
	
	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=5, min_lr=1e-4, max_lr=1e-3)
	adamW_optim = utils.AdamW()
	cnn_final.minibatchGD(X_train=X_train, Y_train=data['Y_train'], y_train=data['y_train'], X_val=X_val, \
					  Y_val=data['Y_val'], y_val=data['y_val'], schedule=cyclic_schedule, optim=adamW_optim, batch_size=200, max_num_epochs=200, show_plots=True)

	cnn_final.save_checkpoint("test_second_minibatch")


def cifar_minibatch():
	"""Test cifar"""

	debug_file = 'Datasets/debug_info.npz'
	load_data = np.load(debug_file)

	X = torch.tensor(load_data['X']).float() # (d, n)

	nf = 2
	width = 32

	loader = dataloader.Dataloader()
	data = loader.load_cifar_data()
	
	n_classes = data['Y_train'].shape[0]
	f = 4
	nf = 100
	m = 100
	rgb = 3

	conv = convolution.Conv_Layer(filter_width=f, nf=nf, channels=rgb, activation_function="ReLu")

	mlp_1 = mlp.MLP_Layer(input_dim=(width//f)**2*nf, output_dim=m, activation_function="ReLu")

	mlp_2 = mlp.MLP_Layer(input_dim=m, output_dim=n_classes, activation_function="softmax")

	cnn_final = cnn.CNN([conv, mlp_1, mlp_2], lam=0.01, dropout_prob=0.4)
	#cnn_final.load_checkpoint("test_first_cifar")
	cnn_final.dataloader = loader
	
	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=5, min_lr=1e-4, max_lr=1e-3)
	adamW_optim = utils.AdamW()
	cnn_final.minibatchGD(X_train=data['X_train'][..., :1000], Y_train=data['Y_train'][..., :1000], y_train=data['y_train'][..., :1000], X_val=data['X_val'], \
					  Y_val=data['Y_val'], y_val=data['y_val'], schedule=cyclic_schedule, optim=adamW_optim, batch_size=200, max_num_epochs=200, show_plots=True)

	cnn_final.save_checkpoint("test_first_cifar")


def time_network():
	"""Time the network. """

	torch.manual_seed(42)

	dataload = dataloader.Dataloader()
	data = dataload.load_train_data(print_progress=True, train_lim=10000, val_lim=1000)
	X_train = data['X_train']
	X_train = dataload.normalize(X_train)
	X_val = dataload.normalize(data['X_val'], use_train_data=True)

	f = 3
	nf = 128
	stride = 1
	padding = 1
	rgb = 3
	n_classes = data['Y_train'].shape[0]
	width = X_train.shape[0]

	conv_1 = convolution.Conv_Layer(filter_width=f, channels=rgb, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	#conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	conv_2 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)

	pool_1 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	conv_3 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	conv_4 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)

	pool_2 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	conv_5 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	pool_3 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	#mlp_1 = mlp.MLP_Layer(input_dim=load_data['conv_flat'].shape[0], output_dim=load_data['X1'].shape[1], activation_function="ReLu")
	mlp_1 = mlp.MLP_Layer(input_dim=(width//2**3)**2*nf, output_dim=400, activation_function="ReLu")

	mlp_2 = mlp.MLP_Layer(input_dim=400, output_dim=400, activation_function="ReLu")

	mlp_3 = mlp.MLP_Layer(input_dim=400, output_dim=n_classes, activation_function="softmax")

	cnn_test = cnn.CNN([conv_1, conv_2, pool_1, conv_3, conv_4, pool_2, conv_5, pool_3, mlp_1, mlp_2, mlp_3], device="mps")


	# Build one real minibatch the same way minibatchGD would
	X_batch = X_train[..., :200].to(cnn_test.device)
	Y_batch = data['Y_train'][:, :200].to(cnn_test.device)

	# Warm-up: let MPS compile/cache kernels for every op we're about to use,
	# outside of the timed measurement.
	for _ in range(3):
		P_batch = cnn_test.forward(X_batch, cnn_test.dropout_prob)
		cost, _ = cnn_test.cost_function(P_batch, Y_batch)
		cost.backward()
		cnn_test.sync()

	prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True)

	for _ in range(30):
		P_batch = cnn_test.forward(X_batch, cnn_test.dropout_prob)
		cost, _ = cnn_test.cost_function(P_batch, Y_batch)

		prof.start()
		cost.backward()
		cnn_test.sync()
		prof.stop()
		

	print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


class LRRangeTestSchedule(utils.lr_schedule):
	"""Ramps LR exponentially from min_lr to max_lr over total_steps updates.
	Throwaway schedule for finding a good min_lr/max_lr — not for real training."""
	def __init__(self, min_lr: float = 1e-7, max_lr: float = 1.0, total_steps: int = 500):
		self.log_min = math.log10(min_lr)
		self.log_max = math.log10(max_lr)
		self.total_steps = total_steps

	def get_state(self):
		return {"log_min": self.log_min, "log_max": self.log_max, "total_steps": self.total_steps}

	def set_state(self, state):
		self.log_min = state["log_min"]
		self.log_max = state["log_max"]
		self.total_steps = state["total_steps"]

	def get_lr(self, iteration):
		frac = min(iteration / self.total_steps, 1.0)
		log_lr = self.log_min + frac * (self.log_max - self.log_min)
		return 10 ** log_lr


def run_lr_range_test(device: str = "mps", train_lim: int = 8000, val_lim: int = 500,
					   batch_size: int = 200, min_lr: float = 1e-7, max_lr: float = 1.0,
					   lam: float = 2.5e-4, dropout_prob: float = 0.3, smoothing_window: int = 15,
					   seed: int = 42) -> dict:
	"""Runs an LR range test on a fresh CNN instance, plots loss vs. LR, and returns
	a suggested min_lr/max_lr along with the raw data for further inspection."""
	torch.manual_seed(seed)

	dataload = dataloader.Dataloader()

	# Subset load — data is pre-shuffled by split_tiny_imagenet's stratified split
	data = dataload.load_train_data(train_lim=train_lim, val_lim=val_lim, print_progress=True)
	X_train = dataload.normalize(data['X_train'])
	del data['X_train']
	X_val = dataload.normalize(data['X_val'], use_train_data=True)
	del data['X_val']
	dataload.set_normal_device(device=device)

	n_classes = data['Y_train'].shape[0]
	width = X_train.shape[0]
	rgb = 3
	f, nf, stride, padding = 3, 128, 1, 1

	conv_1 = convolution.Conv_Layer(filter_width=f, channels=rgb, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	conv_2 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)

	pool_1 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	conv_3 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	conv_4 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)

	pool_2 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	conv_5 = convolution.Conv_Layer(filter_width=f, channels=nf, nf=nf, activation_function="ReLu", stride=stride, padding=padding)
	
	pool_3 = pooling.Pool_Layer(kernel=2, stride=2, padding=0)

	mlp_1 = mlp.MLP_Layer(input_dim=(width//2**3)**2*nf, output_dim=400, activation_function="ReLu")

	mlp_2 = mlp.MLP_Layer(input_dim=400, output_dim=400, activation_function="ReLu")

	mlp_3 = mlp.MLP_Layer(input_dim=400, output_dim=n_classes, activation_function="softmax")

	cnn_lrtest = cnn.CNN([conv_1, conv_2, pool_1, conv_3, conv_4, pool_2, conv_5, pool_3,
						   mlp_1, mlp_2, mlp_3], lam=lam, device=device, dropout_prob=dropout_prob)
	cnn_lrtest.dataloader = dataload

	num_minibatches = train_lim // batch_size
	schedule = LRRangeTestSchedule(min_lr=min_lr, max_lr=max_lr, total_steps=num_minibatches)
	optim = utils.AdamW()

	cnn_lrtest.minibatchGD(
		X_train=X_train, Y_train=data['Y_train'], y_train=data['y_train'],
		X_val=X_val, Y_val=data['Y_val'], y_val=data['y_val'],
		schedule=schedule, optim=optim,
		batch_size=batch_size, max_num_epochs=1,
		shuffle_set=True, show_plots=True, profile=False, debug_logs=False,
	)

	lrs = np.array(cnn_lrtest.history['learning_rates'])
	losses = np.array(cnn_lrtest.history['train_losses'])

	if len(losses) >= smoothing_window:
		kernel = np.ones(smoothing_window) / smoothing_window
		losses_smooth = np.convolve(losses, kernel, mode='valid')
		lrs_smooth = lrs[smoothing_window - 1:]
	else:
		losses_smooth, lrs_smooth = losses, lrs

	log_lrs_smooth = np.log10(lrs_smooth)
	slopes = np.diff(losses_smooth) / np.diff(log_lrs_smooth)
	steepest_idx = np.argmin(slopes)  # most negative slope = fastest loss drop
	suggested_max_lr = lrs_smooth[steepest_idx]
	suggested_min_lr = suggested_max_lr / 10

	plt.figure(figsize=(8, 5))
	plt.plot(lrs, losses, alpha=0.3, label="raw")
	plt.plot(lrs_smooth, losses_smooth, label="smoothed")
	plt.axvline(suggested_max_lr, color='r', linestyle='--', label=f"suggested max_lr ≈ {suggested_max_lr:.2e}")
	plt.axvline(suggested_min_lr, color='g', linestyle='--', label=f"suggested min_lr ≈ {suggested_min_lr:.2e}")
	plt.xscale('log')
	plt.xlabel("Learning rate")
	plt.ylabel("Training loss (cross-entropy)")
	plt.title("LR range test")
	plt.legend()
	plt.show()

	print(f"Suggested min_lr: {suggested_min_lr:.2e}")
	print(f"Suggested max_lr: {suggested_max_lr:.2e}")
	print("Check the plot before trusting these — pick max_lr just before the curve flattens or turns upward.")

	return {
		"lrs": lrs, "losses": losses,
		"lrs_smooth": lrs_smooth, "losses_smooth": losses_smooth,
		"suggested_min_lr": suggested_min_lr, "suggested_max_lr": suggested_max_lr,
	}


def big_cifar():
	"""Test supervised learning of a big model with CIFAR-10. """
	torch.manual_seed(42)

	device = "mps"

	dataload = dataloader.Dataloader()
	data = dataload.load_cifar_data()

	X_train = dataload.normalize(data['X_train'][..., :5000])
	del data['X_train']
	X_val = dataload.normalize(data['X_val'], use_train_data=True)
	del data['X_val']
	dataload.set_normal_device(device=device)

	batch_size = 200
	n_train = X_train.shape[-1]

	f = 3
	stride = 1
	padding = 1
	rgb = 3
	lam = 2.5e-4
	n_classes = data['Y_train'].shape[0]
	width = X_train.shape[0]

	conv_1 = convolution.Conv_Layer(filter_width=f, channels=rgb, nf=32, activation_function="ReLu", stride=stride, padding=padding)
	#conv.F = Fs.reshape((f*f*3, nf), order='C') # (f^2*rgb, nf)

	conv_2 = convolution.Conv_Layer(filter_width=f, channels=32, nf=32, activation_function="ReLu", stride=stride, padding=padding)

	pool_1 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0.3)

	conv_3 = convolution.Conv_Layer(filter_width=f, channels=32, nf=64, activation_function="ReLu", stride=stride, padding=padding)
	
	conv_4 = convolution.Conv_Layer(filter_width=f, channels=64, nf=64, activation_function="ReLu", stride=stride, padding=padding)

	pool_2 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0.5)

	conv_5 = convolution.Conv_Layer(filter_width=f, channels=64, nf=128, activation_function="ReLu", stride=stride, padding=padding)

	conv_6 = convolution.Conv_Layer(filter_width=f, channels=128, nf=128, activation_function="ReLu", stride=stride, padding=padding)
	
	pool_3 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0.5)

	mlp_1 = mlp.MLP_Layer(input_dim=(width//2**3)**2*128, output_dim=128, activation_function="ReLu", dropout_prob=0.5)

	mlp_2 = mlp.MLP_Layer(input_dim=128, output_dim=n_classes, activation_function="linear")

	cnn_test = cnn.CNN([conv_1, conv_2, pool_1, conv_3, conv_4, pool_2, conv_5, conv_6, pool_3, mlp_1, mlp_2], \
					lam=lam, device=device)
	
	cnn_test.dataloader = dataload

	k = 2
	iters_per_epoch = n_train / batch_size
	starting_cycle_length = k * iters_per_epoch    # k in [2, 10]
	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=starting_cycle_length, min_lr=1e-4, max_lr=1e-3)
	adamW_optim = utils.AdamW()
	cnn_test.minibatchGD(X_train=X_train, Y_train=data['Y_train'][..., :5000], y_train=data['y_train'][:5000], X_val=X_val, \
						Y_val=data['Y_val'], y_val=data['y_val'], schedule=cyclic_schedule, optim=adamW_optim, \
						batch_size=batch_size, max_num_epochs=200, show_plots=True, profile=False, debug_logs=False)



if __name__ == "__main__":
	#test_train_contrastive_tiny(checkpoint=None)
	#time_network()
	big_cifar()
	#test_overfit_small_sample()
