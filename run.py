import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import mlp, cnn, convolution, dataloader, pooling
import utils

class Settings:
	def __init__(self, training_samples = 1000, validation_samples = 1000, batch_size=100, show_plots: bool = True, dropout_prob: float = 0):
		self.training_samples = training_samples
		self.validation_samples = validation_samples
		self.batch_size = batch_size
		self.show_plots = show_plots
		self.dropout_prob = dropout_prob

def run_contrastive(settings: Settings = Settings(), checkpoint: str = None, contrastive_epochs: int = 20, epoch_save_step: int = 5):
	"""Runs contrastive learning on the full dataset."""
	torch.manual_seed(42)

	device = "mps"
	dataload = dataloader.Dataloader()
	data = dataload.load_cifar_data()
	X_train = dataload.normalize(data['X_train'][..., :settings.training_samples])
	del data['X_train']
	X_val = dataload.normalize(data['X_val'][..., :settings.validation_samples], use_train_data=True)
	del data['X_val']
	dataload.set_normal_device(device=device)

	

	f = 3
	stride = 1
	padding = 1
	rgb = 3
	lam = 1e-2
	n_classes = data['Y_train'].shape[0]
	width = X_train.shape[0]

	conv_1 = convolution.Conv_Layer(filter_width=f, channels=rgb, nf=32, activation_function="ReLu", stride=stride, padding=padding)

	conv_2 = convolution.Conv_Layer(filter_width=f, channels=32, nf=32, activation_function="ReLu", stride=stride, padding=padding)

	pool_1 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0)

	conv_3 = convolution.Conv_Layer(filter_width=f, channels=32, nf=64, activation_function="ReLu", stride=stride, padding=padding)
	
	conv_4 = convolution.Conv_Layer(filter_width=f, channels=64, nf=64, activation_function="ReLu", stride=stride, padding=padding)

	pool_2 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0)

	conv_5 = convolution.Conv_Layer(filter_width=f, channels=64, nf=128, activation_function="ReLu", stride=stride, padding=padding)

	conv_6 = convolution.Conv_Layer(filter_width=f, channels=128, nf=128, activation_function="ReLu", stride=stride, padding=padding)
	
	pool_3 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0)

	mlp_1 = mlp.MLP_Layer(input_dim=(width//2**3)**2*128, output_dim=128, activation_function="ReLu", dropout_prob=0)

	mlp_2 = mlp.MLP_Layer(input_dim=128, output_dim=n_classes, activation_function="linear")

	n_train = X_train.shape[-1]
	k = 2
	iters_per_epoch = n_train / settings.batch_size
	starting_cycle_length = k * iters_per_epoch    # k in [2, 10]
	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=starting_cycle_length, min_lr=5e-3, max_lr=5e-2)
	adamW_optim = utils.AdamW()

	cnn_test = cnn.CNN([conv_1, conv_2, pool_1, conv_3, conv_4, pool_2, conv_5, conv_6, pool_3, mlp_1, mlp_2], \
					lam=lam, device=device, lr_schedule=cyclic_schedule, adamW_optim=adamW_optim)
	
	cnn_test.dataloader = dataload
	

	if checkpoint: 
		try:
			cnn_test.load_checkpoint(checkpoint)
		except:
			print(f"Could not find checkpoint with filename {checkpoint}. Starting from scratch.")


	try: 
		cnn_test.contrastiveLearning(X_train=X_train, X_val=X_val, \
					batch_size=settings.batch_size, max_num_epochs=contrastive_epochs, epoch_save_step=epoch_save_step, show_plots=True, debug_logs=False, \
						save_path=checkpoint, normalize_at=[])
		cnn_test.save_checkpoint(f"{checkpoint}_contrastive", include_training_state=False) if checkpoint else None
	except KeyboardInterrupt:
		print("Training interupted, saving snapshot...")
		if checkpoint: cnn_test.save_checkpoint(f"{checkpoint}_contrastive_interupted", include_training_state=False)


def start_finetuning(settings: Settings = Settings(), checkpoint: str = None, epochs: int = 20, epoch_save_step: int = 5, freeze_body_epochs: int = 10):
	"""Starts supervised finetuning on the model after contrastive learning."""
	torch.manual_seed(42)

	device = "mps"
	dataload = dataloader.Dataloader()
	data = dataload.load_cifar_data()
	X_train = dataload.normalize(data['X_train'][..., :settings.training_samples])
	del data['X_train']
	X_val = dataload.normalize(data['X_val'][..., :settings.validation_samples], use_train_data=True)
	del data['X_val']
	dataload.set_normal_device(device=device)

	Y_train=data['Y_train']
	y_train = data['y_train']
	
	f = 3
	stride = 1
	padding = 1
	rgb = 3
	lam = 2.5e-4
	n_classes = data['Y_train'].shape[0]
	width = X_train.shape[0]

	# Body inherited from contrastive ================== 
	conv_1 = convolution.Conv_Layer(filter_width=f, channels=rgb, nf=32, activation_function="ReLu", stride=stride, padding=padding)

	conv_2 = convolution.Conv_Layer(filter_width=f, channels=32, nf=32, activation_function="ReLu", stride=stride, padding=padding)

	pool_1 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0)

	conv_3 = convolution.Conv_Layer(filter_width=f, channels=32, nf=64, activation_function="ReLu", stride=stride, padding=padding)
	
	conv_4 = convolution.Conv_Layer(filter_width=f, channels=64, nf=64, activation_function="ReLu", stride=stride, padding=padding)

	pool_2 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0)

	conv_5 = convolution.Conv_Layer(filter_width=f, channels=64, nf=128, activation_function="ReLu", stride=stride, padding=padding)

	conv_6 = convolution.Conv_Layer(filter_width=f, channels=128, nf=128, activation_function="ReLu", stride=stride, padding=padding)
	
	pool_3 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0)

	# Finetuning head =====================================
	mlp_1 = mlp.MLP_Layer(input_dim=(width//2**3)**2*128, output_dim=128, activation_function="ReLu", dropout_prob=0.5)
	
	mlp_2 = mlp.MLP_Layer(input_dim=128, output_dim=n_classes, activation_function="linear")

	n_train = X_train.shape[-1]
	k = 2
	iters_per_epoch = n_train / settings.batch_size
	starting_cycle_length = k * iters_per_epoch    # k in [2, 10]
	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=starting_cycle_length, min_lr=1e-4, max_lr=1e-3)
	adamW_optim = utils.AdamW()

	cnn_test = cnn.CNN([conv_1, conv_2, pool_1, conv_3, conv_4, pool_2, conv_5, conv_6, pool_3, mlp_1, mlp_2], \
					lam=lam, device=device, lr_schedule=cyclic_schedule, adamW_optim=adamW_optim)
	
	
	cnn_test.dataloader = dataload

	if checkpoint: 
		try:
			cnn_test.load_checkpoint(checkpoint, load_training_state=False, reinitialize_last=3)
			print("Successfully loaded checkpoint for finetuning.")
		except:
			print(f"Could not find checkpoint with filename {checkpoint}. Starting from scratch.")

	try: 
		# Body will be frozen for the first
		cnn_test.minibatchGD(X_train=X_train, Y_train=Y_train, y_train=y_train, X_val=X_val, \
					Y_val=data['Y_val'], y_val=data['y_val'], schedule=cyclic_schedule, optim=adamW_optim, \
					batch_size=settings.batch_size, max_num_epochs=epochs, show_plots=True, profile=True, debug_logs=False, save_path=checkpoint, epoch_save_step=epoch_save_step, \
						supervised_head_size=2, freeze_body_epochs=freeze_body_epochs, normalize_at=[8])

		if checkpoint: cnn_test.save_checkpoint(f"{checkpoint}_supervised")
	except KeyboardInterrupt:
		print("Training interupted, saving snapshot...")
		if checkpoint: cnn_test.save_checkpoint(f"{checkpoint}_supervised_interupted")


def run_pure_supervised(settings: Settings = Settings(), checkpoint: str = None, supervised_epochs: int = 20, epoch_save_step: int = 5):
	"""Runs contrastive learning on the full dataset."""
	
	device = "mps"
	dataload = dataloader.Dataloader()
	data = dataload.load_cifar_data()
	X_train = dataload.normalize(data['X_train'][..., :settings.training_samples])
	del data['X_train']
	X_val = dataload.normalize(data['X_val'][..., :settings.validation_samples], use_train_data=True)
	del data['X_val']
	dataload.set_normal_device(device=device)

	Y_train=data['Y_train']
	y_train = data['y_train']
	
	f = 3
	stride = 1
	padding = 1
	rgb = 3
	lam = 2.5e-4
	n_classes = data['Y_train'].shape[0]
	width = X_train.shape[0]

	# Body is not given by contrastive ================== 
	conv_1 = convolution.Conv_Layer(filter_width=f, channels=rgb, nf=32, activation_function="ReLu", stride=stride, padding=padding)

	conv_2 = convolution.Conv_Layer(filter_width=f, channels=32, nf=32, activation_function="ReLu", stride=stride, padding=padding)

	pool_1 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0)

	conv_3 = convolution.Conv_Layer(filter_width=f, channels=32, nf=64, activation_function="ReLu", stride=stride, padding=padding)
	
	conv_4 = convolution.Conv_Layer(filter_width=f, channels=64, nf=64, activation_function="ReLu", stride=stride, padding=padding)

	pool_2 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0)

	conv_5 = convolution.Conv_Layer(filter_width=f, channels=64, nf=128, activation_function="ReLu", stride=stride, padding=padding)

	conv_6 = convolution.Conv_Layer(filter_width=f, channels=128, nf=128, activation_function="ReLu", stride=stride, padding=padding)
	
	pool_3 = pooling.Pool_Layer(kernel=2, stride=2, padding=0, dropout_prob=0)

	mlp_1 = mlp.MLP_Layer(input_dim=(width//2**3)**2*128, output_dim=128, activation_function="ReLu", dropout_prob=0)
	
	mlp_2 = mlp.MLP_Layer(input_dim=128, output_dim=n_classes, activation_function="linear")

	n_train = X_train.shape[-1]
	k = 2
	iters_per_epoch = n_train / settings.batch_size
	starting_cycle_length = k * iters_per_epoch    # k in [2, 10]
	cyclic_schedule = utils.cyclic_increasing_lr(starting_cycle_length=starting_cycle_length, min_lr=5e-3, max_lr=5e-2)
	adamW_optim = utils.AdamW()

	cnn_test = cnn.CNN([conv_1, conv_2, pool_1, conv_3, conv_4, pool_2, conv_5, conv_6, pool_3, mlp_1, mlp_2], \
					lam=lam, device=device, lr_schedule=cyclic_schedule, adamW_optim=adamW_optim)
	
	
	cnn_test.dataloader = dataload


	if checkpoint: 
		try:
			cnn_test.load_checkpoint(checkpoint, load_training_state=False)
		except:
			print(f"Could not find checkpoint with filename {checkpoint}. Starting from scratch.")

	try: 
		cnn_test.minibatchGD(X_train=X_train, Y_train=Y_train, y_train=y_train, X_val=X_val, \
					Y_val=data['Y_val'], y_val=data['y_val'], schedule=cyclic_schedule, optim=adamW_optim, \
					batch_size=settings.batch_size, max_num_epochs=supervised_epochs, show_plots=True, profile=True, debug_logs=False, save_path=checkpoint, epoch_save_step=epoch_save_step, \
						normalize_at=[7])

		if checkpoint: cnn_test.save_checkpoint(f"{checkpoint}_supervised")
	except KeyboardInterrupt:
		print("Training interupted, saving snapshot...")
		if checkpoint: cnn_test.save_checkpoint(f"{checkpoint}_supervised_interupted")



def run_full(settings: Settings = Settings(), checkpoint: str = None, contrastive_epochs: int = 50, supervised_epochs: int = 20):
	"""Runs both contrastive learning and supervised training on the full dataset."""
	run_contrastive(settings, checkpoint=checkpoint, contrastive_epochs=contrastive_epochs, epoch_save_step=5)
	start_finetuning(settings, checkpoint=checkpoint, epochs=10, epoch_save_step=5, freeze_body_epochs=5)

if __name__ == "__main__":
	# In the CIFAR set, there are 60000 training samples and 1000 validation samples.
	settings = Settings(training_samples=49000, validation_samples=1000, batch_size=100, show_plots=True, dropout_prob=0)

	checkpoint= "CIFAR_FIRST_contrastive_interupted"
	#run_contrastive(settings, checkpoint=checkpoint, contrastive_epochs=60, epoch_save_step=5)
	start_finetuning(settings, checkpoint=checkpoint, epochs=50, epoch_save_step=2, freeze_body_epochs=0)
	#run_pure_supervised(settings, checkpoint=checkpoint, supervised_epochs=20, epoch_save_step=5)
