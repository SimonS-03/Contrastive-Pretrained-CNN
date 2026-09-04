import matplotlib.pyplot as plt
import torch
import tqdm
from dataloader import Dataloader
from convolution import Conv_Layer
from pooling import Pool_Layer
from mlp import MLP_Layer
from utils import *
import numpy as np
import time

class CNN:
	def __init__(self, layers: list[Conv_Layer | MLP_Layer | Pool_Layer], lam: float = 1e-3, device:str="cpu", adamW_optim: optimizer = None, lr_schedule: lr_schedule = None):
		"""Creates a linear classifier with 10 output classes.
		Args:
			layers (list[Conv_Layer | MLP_Layer | Pool_Layer]): A list of either convolution-layers or mlp-layers, already initialized.
			lam (float): The lambda used in L2 regularization of the layers during backward pass. 
			devide (str): Which device to run the layer on. (cpu, mps)"""
		self.layers = layers
		"""All the neural network layers. """
		self.lam = lam
		"""The regularisation term. """
		self.dataloader = Dataloader()
		"""The dataloader class for this CNN. """
		self.device:str = device
		"""The device that the CNN runs on. """
		self.adamW_optim = adamW_optim
		self.lr_schedule = lr_schedule

		self.history = {
			'epochs': [], 'train_steps': [], 'train_costs': [], 'train_losses': [],
			'val_steps': [], 'val_costs': [], 'val_losses': [],
			'train_accuracys': [], 'val_accuracys': [], 'learning_rates': []
		}
		"""The history of the training run. \\
			'epochs': [], 'train_steps': [], 'train_costs': [], 'train_losses': [],
					'val_steps': [], 'val_costs': [], 'val_losses': [],
					'train_accuracys': [], 'val_accuracys': [], 'learning_rates': []"""
	
		self.current_epoch = 0
		"""Current epoch. """

		self.__to(device)


	def __to(self, device:str) -> None:
		"""Sets all the layers to the correct device. 
		
		Args:
			device (str): The device to set the layers to. (cpu, mps)"""

		for layer in self.layers:
			if isinstance(layer, Pool_Layer) or isinstance(layer, Conv_Layer):
				layer.device = device
				layer.nn_layer = layer.nn_layer.to(device=device)

			if isinstance(layer, Conv_Layer):
				layer.W = layer.nn_layer.weight
				layer.b = layer.nn_layer.bias

			if isinstance(layer, MLP_Layer):
				layer.W = layer.W.detach().to(device).requires_grad_()
				layer.b = layer.b.detach().to(device).requires_grad_()


	def sync(self) -> None:
		"""Syncronizes the code. """

		if self.device == "mps" and torch.backends.mps.is_available():
			torch.mps.synchronize()
		elif self.device == "cuda" and torch.cuda.is_available():
			torch.cuda.synchronize()
	

	def forward(self, x: torch.Tensor,  apply_dropout:bool=True, normalize_at: list[int] = []) -> torch.Tensor:
		"""Do a forward pass of the classifier with the loaded Mx (needs to be loaded!)
		
		Args:
			x (torch.Tensor): The input of the whole network. (height, width, nf, n)
			apply_dropout (bool): If dropout shall be applied. 
			
		Returns:
			x (torch.Tensor): The output of the whole network. (output_dim, n)"""

		prev_layer = None
		for i, layer in enumerate(self.layers):

			if prev_layer == None and (isinstance(layer, Conv_Layer) or isinstance(layer, Pool_Layer)):
				x = x.permute(3, 2, 0, 1).contiguous() # (n, nf, h, w)

			if prev_layer != None and not isinstance(prev_layer, MLP_Layer) and isinstance(layer, MLP_Layer): # If MLP and prev layer is not MLP

				#Permute back
				x = x.permute(2, 3, 1, 0).contiguous() # (h, w, nf, n)
			
				#Flatten input
				x = x.reshape(x.shape[0]*x.shape[1]*x.shape[2], x.shape[3])
			
			x = layer.forward(x, apply_dropout=apply_dropout)

			# Normalization 
			if normalize_at and i in normalize_at:
				mean = x.mean(dim=0, keepdim=True)
				var = x.var(dim=0, keepdim=True, unbiased=False)
				x = (x - mean) / torch.sqrt(var + 1e-5)

			prev_layer = layer
			
		return x


	def cost_function(self, logits: torch.Tensor, Y_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Calculated the mean cross-entropy loss of a set of 
		predicted class probabilites"""
		# both have dimensions (10, batch_size)
		# We take the mean of the loss over the batch
		#targets = torch.argmax(Y_batch, dim=0)
		targets = torch.argmax(Y_batch, dim=0)
		cross_entropy = torch.nn.functional.cross_entropy(logits.T, targets)
		reg = 0
		if self.lam != 0: #Reg is zero if lam is zero
			for layer in self.__param_layers:
				cur_weights:torch.Tensor = layer.W
				reg += self.lam * torch.sum(cur_weights**2)

		return cross_entropy + reg, cross_entropy


	def __cross_entropy_loss(self, p: torch.Tensor, Y_batch: torch.Tensor, eps:float=1e-10) -> torch.Tensor:
		"""Calculates the cross entropy loss. 
		
		Args:
			p (torch.Tensor): The calculated probabilites. (n_classes, n)
			Y_batch (torch.Tensor): The actual one-hot labels. (n_classes, n)
			eps (float): Small number to avoid overflow errors when doing log(p). 
			
		Returns:
			yp_mean (torch.Tensor): The cross entropy loss. (1)"""

		Y_P = -Y_batch*torch.log(p+eps) # (n_classes, n)

		yp_sum = torch.sum(Y_P, axis = 0) # (n)

		yp_mean = torch.mean(yp_sum) # (1)

		return yp_mean


	def get_gradient(self, Mx_batch:torch.Tensor, Y_batch:torch.Tensor, P_batch:torch.Tensor) -> list[dict]:
		"""Returns the gradient of the loss wrt weights and biases in a tuple. Will use self.s1 and self.h1
		
		Args:
			Mx_batch (torch.Tensor): The input data batch which was used for the current training batch. (n_filters, filter_size, batch_size)
			Y_batch (torch.Tensor): The output/class data batch which was used for the current training batch. (num_labels, batch_size)
			P_batch (torch.Tensor): The prediction probabilities for the current batch. (num_labels, batch_size)

		Returns:
			grads (list[dict]): List of dictionaries of the gradients of the weights and biases for each layer. \
				Sorted from first to last layer. 
			
			Each dictionary in the list has the gradient of the weights (either **'W'** or **'F'**) and bias of one layer:
				* **'W'** (torch.Tensor): Gradient of the weights in the layer. (local_output_dim, local_input_dim)
				* **'b'** (torch.Tensor): Gradient of the bias in one layer. (local_output_dim, 1)"""
		
		g_batch = -(Y_batch-P_batch).T # (batch_size, num_labels)
		batch_size = g_batch.shape[0] # nbr points in batch
		
		grads = []
		for i in range(1, len(self.layers)+1):
			layer = self.layers[-i]
			is_input_layer = i == len(self.layers)
			
			layer_input = None
			if is_input_layer: layer_input = Mx_batch
			else: layer_input = self.layers[-(i+1)].h

			# Activation function reversal
			if layer.activation_function == "ReLu":
				g_batch = g_batch * (layer.h > 0).T / (1 - self.dropout_prob)

			# Gradient calculation 
			if isinstance(layer, Conv_Layer):
				g_batch = g_batch.T.reshape(layer.n_p, layer.nf, batch_size)
				g_sum = torch.sum(g_batch, axis=(0, 2)).reshape((layer.nf, 1)) /batch_size
				layer_input_T = layer_input.permute(1, 0, 2)
				dl_dW = torch.einsum('ijn, jln ->il', layer_input_T, g_batch)/batch_size


			elif isinstance(layer, MLP_Layer):
				dl_dW = (g_batch.T @ layer_input.T)/batch_size
				g_sum = g_batch.sum(axis = 0).unsqueeze(dim = 1) / batch_size

			grad = {}
			grad["W"] = dl_dW
			grad["b"] = g_sum
			grads.append(grad)

			# Weight reversal 
			if isinstance(layer, Conv_Layer):
				pass
			elif isinstance(layer, MLP_Layer):
				g_batch = g_batch @ layer.W
				g_sum = g_batch.sum(axis = 0).unsqueeze(dim = 1)
		
		grads.reverse()
		return grads


	def __back_pass(self, lr: float, grads: list[dict[str, float]]) -> None:
		"""Updates the network based on the gradients. 
		
		Args:
			lr: learning rate,
			grads: list of dictionaries containign gradient for each parameter.
			"""

		for i in range(len(grads)):
			cur_layer = self.__param_layers[i]
			with torch.no_grad():
				cur_layer.W.data -= lr * grads[i]['W']
				cur_layer.b.data -= lr * grads[i]['b']
				#print(f"Mag of W after back pass: {cur_layer.W.norm()}")
				#print(f"Mag of b after back pass: {cur_layer.b.norm()}")


	@property
	def __param_layers(self) -> list:
		"""List of the layers that have tunable parameters (all except the pooling layers). """

		return [l for l in self.layers if not isinstance(l, Pool_Layer)]


	def __average_loss(self, list_to_average:list, surrounding_items:int):
		"""Return the average of the list given how many steps back to average. 
		
		list_to_average (list): The list to average over. 
		surrounding_items (int): The number of items around to average. Should be an even number. """

		if surrounding_items % 2 != 0:
			raise ValueError("surrounding_items in __average loss needs to be even. ")

		average_list = []

		lookback_steps = surrounding_items // 2
		lookahead_steps = surrounding_items // 2

		list_len = len(list_to_average)

		for i in range(list_len):
			cur_sum = 0

			cur_lookback = min(i, lookback_steps)          # how many steps back (not counting i)
			for j in range(cur_lookback + 1):               # includes i itself once (j=0)
				cur_sum += list_to_average[i - j]

			cur_lookahead = min(list_len - 1 - i, lookahead_steps)  # how many steps forward
			for k in range(1, cur_lookahead + 1):            # starts at 1, so i is NOT re-added
				cur_sum += list_to_average[i + k]

			count = cur_lookback + 1 + cur_lookahead
			average_list.append(cur_sum / count)

		return average_list


	def minibatchGD(self, X_train:torch.Tensor, Y_train:torch.Tensor, y_train:torch.Tensor, X_val:torch.Tensor, Y_val:torch.Tensor, 
				 y_val:torch.Tensor, schedule: lr_schedule, optim: optimizer, max_num_epochs: int = 40, batch_size: int = 100, 
					shuffle_set: bool = True, show_plots: bool = True, epoch_plot_period: int = 1, profile:bool=False, 
					save_path: str = None, epoch_save_step: int = 5, profile_batches:int=30, profile_warmup_batches:int=5, debug_logs:bool=False, supervised_head_size: int = 0, freeze_body_epochs: int = 0,
					normalize_at: list[int] = []):
		""" Does end-to-end minibatch gradient descent with the given training data. 
		
		Args:
			X_train (torch.Tensor): The training data. (height, width, rgb, n_train)
			Y_train (torch.Tensor): The training labels as one hot vectors. (n_classes, n_train)
			y_train (torch.Tensor): The training labels. (n_train,)
			X_val (torch.Tensor): The validation data. (height, width, rgb, n_val)
			Y_val (torch.Tensor): The validation labels as one hot vectors. (n_classes, n_val)
			y_val (torch.Tensor): The validation labels. (n_val,)
			schedule (lr_schedule): The learning rate schedule. 
			optim (optimizer): The optimizer. 
			max_num_epochs (int): Maximum number of epochs for the training. 
			batch_size (int): The batch size for batch training. 
			shuffle_set (bool): If the data set should be randomly shuffled at before each epoch. 
			show_plots (bool): If plots of the training and validation loss, cost and accuracy should be displayed at the end. 
			profile_batches (int): How many batches to time. 
			profile_warmup_batches (int): How many batches to run before starting the timing. To avoid timing one-time things. 
			debug_logs (bool): If debugs should be calculated and printed. """

		Y_train = Y_train.to(self.device)
		Y_val = Y_val.to(self.device)
		y_train = y_train.to(self.device)
		y_val = y_val.to(self.device)

		n_datapoints = X_train.shape[3]
		num_minibatches = int(n_datapoints / batch_size)

		val_cost = torch.inf
		
		if show_plots:
			plt.ion() # Turn on interactive mode

			# Single window, three subplots: cost/loss, accuracy, learning rate
			fig, (ax1, ax2, lr_ax) = plt.subplots(1, 3, figsize=(18, 5))
			fig.canvas.manager.set_window_title("Training metrics")

			lr_line, = lr_ax.plot([], [], linestyle='-', color='blue')
			lr_ax.set_xlabel("Epoch")
			lr_ax.set_title("Learning Rate")
			lr_ax.legend()

			fig.show()

			# Filters get their own separate window
			self.show_filters(self.layers[0].W, freeze=False)

		pbar = tqdm.tqdm(range(max_num_epochs))
		update_step = 1

		timings = {"data_slice": 0.0, "augment": 0.0, "forward": 0.0, "cost": 0.0, "backward": 0.0, "optim_step": 0.0, "param_update": 0.0}
		profiled_batches_done = 0
		batches_seen = 0

		for i in pbar:
			#print(f"Started epoch {i}")
			self.current_epoch += 1

			#Get the permutation of this epoch
			permutation = torch.randperm(n_datapoints) if shuffle_set else torch.arange(n_datapoints)

			if supervised_head_size > 0 and i < freeze_body_epochs: # Keeps body frozen for first few epochs
				for layer in self.__param_layers[:-supervised_head_size]:
					layer.W.requires_grad = False
					layer.b.requires_grad = False
			else:
				for layer in self.__param_layers:
					layer.W.requires_grad = True
					layer.b.requires_grad = True
				
			for j in range(num_minibatches):
				if debug_logs:	
					print(f"minibatch {j}")

				do_profile = profile and profiled_batches_done < profile_batches \
					and profile_warmup_batches <= batches_seen
				if do_profile:
					self.sync(); t_start = time.perf_counter()

				batch_start = j*batch_size
				batch_end = (j+1)*batch_size

				#Used to get the batch indexes directly without shuffling the whole dataset
				batch_idx = permutation[batch_start:batch_end]

				X_batch = X_train[..., batch_idx]
				X_batch = X_batch.to(self.device)

				if do_profile:
					self.sync(); t = time.perf_counter(); timings["data_slice"] += t - t_start; t_start = t

				with torch.no_grad():
					X_aug = self.dataloader.augment_data(X=X_batch, individual_aug=False, min_crop=.65)

				if do_profile:
					self.sync(); t = time.perf_counter(); timings["augment"] += t - t_start; t_start = t
					
				P_batch = self.forward(X_aug)
				if do_profile:
					self.sync(); t = time.perf_counter(); timings["forward"] += t - t_start; t_start = t
				Y_batch = Y_train[:, batch_idx]
				batch_cost, batch_loss = self.cost_function(P_batch, Y_batch)

				if show_plots:
					self.history['train_steps'].append(update_step / num_minibatches)
					self.history['train_costs'].append(batch_cost.item())
					self.history['train_losses'].append(batch_loss.item())

					ax1.clear()
					ax1.plot(self.history['train_steps'], self.__average_loss(self.history['train_costs'], num_minibatches-1), label="Training Cost", color="y")
					ax1.plot(self.history['train_steps'], self.__average_loss(self.history['train_losses'], num_minibatches-1), label="Training Loss", linestyle='--', color="r")
					ax1.plot(self.history['val_steps'], self.history['val_costs'], label="Validation Cost", color="b")
					ax1.plot(self.history['val_steps'], self.history['val_losses'], label="Validation Loss", linestyle='--', color="c")
					ax1.set_xlabel("Epoch")
					ax1.legend()
					ax1.set_title(f"Train/val cost + loss, lam={self.lam}")
					fig.canvas.draw()
					fig.canvas.flush_events()

				if do_profile:
					self.sync(); t = time.perf_counter(); timings["cost"] += t - t_start; t_start = t

				lr = schedule.get_lr(update_step)
				params:dict[str, torch.Tensor] = [{"W": layer.W, "b": layer.b} for layer in self.__param_layers]

				for param_set in params: # Zeroing gradients before loss calculation
					param_set["W"].grad = None
					param_set["b"].grad = None

				batch_loss.backward()
				if do_profile:
					self.sync(); t = time.perf_counter(); timings["backward"] += t - t_start; t_start = t
				with torch.no_grad():
					auto_grads:list[dict[str, float]] = [{"W": layer.W.grad, "b": layer.b.grad} for layer in self.__param_layers]
					if debug_logs and auto_grads is not None and len(auto_grads) > 0 and auto_grads[0]["W"] is not None:
						first_layer = self.__param_layers[0]
						last_layer = self.__param_layers[-1]
						first_grads = auto_grads[0]
						last_grads = auto_grads[-1]
						print(
							f"\n======= Gradient debug @ update {update_step} =======\n"
							f"First layer ({first_layer.__class__.__name__}) | W grad norm: {first_grads['W'].norm().item():.6f}, "
							f"b grad norm: {first_grads['b'].norm().item():.6f}\n"
							f"Last layer ({last_layer.__class__.__name__}) | W grad norm: {last_grads['W'].norm().item():.6f}, "
							f"b grad norm: {last_grads['b'].norm().item():.6f}\n"
							f"===========================================\n"
						)
					optim_grads = optim.update_gradient(params=params, grads=auto_grads, iteration=update_step, lam=self.lam) 
					if do_profile:
						self.sync(); t = time.perf_counter(); timings["optim_step"] += t - t_start; t_start = t
			
					self.__back_pass(lr, optim_grads)

				if show_plots:
						# Learning rate subplot
						self.history['learning_rates'].append(lr)
						lr_steps = self.history['train_steps']
						lr_line.set_data(lr_steps, self.history['learning_rates'])
						lr_ax.relim()
						lr_ax.autoscale_view()
						fig.canvas.draw()
						fig.canvas.flush_events()

			


				if do_profile:
					self.sync(); t = time.perf_counter(); timings["param_update"] += t - t_start; t_start = t


				# REMOVED OUR OWN GRADIENT CALCULATION
				#grads = self.get_gradient(Mx_batch, Y_batch, P_batch)
				#optim_grads = optim.update_gradient(params=params, grads=grads, iteration=update_step, lam=self.lam) 
				#self.__back_pass(lr, optim_grads)
				
				batches_seen += 1
				update_step += 1

				if do_profile:
					profiled_batches_done += 1
					if profiled_batches_done == profile_batches:
						total = sum(timings.values())
						print(f"\n--- Profile over {profile_batches} minibatches (device={self.device}) ---")
						for phase, secs in timings.items():
							per_batch_ms = 1000 * secs / profile_batches
							pct = 100 * secs / total if total > 0 else 0
							print(f"  {phase:14s}: {per_batch_ms:7.1f} ms/batch  ({pct:4.1f}%)")
						print(f"  {'TOTAL':14s}: {1000*total/profile_batches:7.1f} ms/batch\n")
			

			if not show_plots or (i%epoch_plot_period != 0 and i != max_num_epochs-1): # We only need the lists below for plots
				continue
			
			with torch.no_grad():
				# Filter display window (separate from fig)
				self.show_filters(self.layers[0].W, freeze=False, epoch=self.history['train_steps'][-1])

				self.history['epochs'].append(self.current_epoch)
				self.history['val_steps'].append(update_step/num_minibatches)

				val_loss = torch.tensor([0.0]).to(self.device)
				val_cost = torch.tensor([0.0]).to(self.device)
				val_minibatches = int(X_val.shape[-1] / batch_size)
				for k in range(val_minibatches):
					val_batch_start = k*batch_size
					val_batch_end = (k+1)*batch_size
					X_val_batch = X_val[..., val_batch_start:val_batch_end]
					X_val_batch = X_val_batch.to(self.device)
					Y_val_batch = Y_val[..., val_batch_start:val_batch_end]
					P_val = self.forward(X_val_batch)
					val_cost_cur, val_loss_cur = self.cost_function(P_val, Y_val_batch)
					val_cost += val_cost_cur; val_loss += val_loss_cur

				self.history['val_costs'].append(val_cost.detach().item()/val_minibatches)
				self.history['val_losses'].append(val_loss.detach().item()/val_minibatches)
			
				if X_val is not None and y_val is not None and y_train is not None:
					self.history['val_accuracys'].append(self.get_accuracy(X_val, y_val, batch_size).item())
					self.history['train_accuracys'].append(self.get_accuracy(X_train, y_train, batch_size).item())

				if save_path is not None and (self.current_epoch % epoch_save_step == 0 or self.current_epoch == max_num_epochs):
					self.save_checkpoint(f"{save_path}_epoch_{self.current_epoch}")

			pbar.set_postfix(loss = f"{round(val_loss.item(), 4)}")

			ax2.clear()
			ax2.set_title("Train/Val accuracy over epochs")
			ax2.plot(self.history['epochs'], self.history['val_accuracys'], label="validation")
			ax2.plot(self.history['epochs'], self.history['train_accuracys'], label="training")
			ax2.set_xlabel("Epoch")
			ax2.legend()

			fig.canvas.draw()
			fig.canvas.flush_events()

		if show_plots:
			plt.ioff() # Turn off interactive mode
			plt.show() # Keep the final plot window open when training finishes

	def contrastiveLearning(self, X_train: torch.Tensor, X_val: torch.Tensor,
					 max_num_epochs: int = 40, batch_size: int = 100, epoch_save_step: int = 5,
					shuffle_set: bool = True, show_plots: bool = True, debug_logs: bool = False, save_path: str = None, normalize_at: list[int] = []):
		""" Does contrastive self-supervised learning """

		n_datapoints = X_train.shape[-1]
		num_minibatches = int(n_datapoints / batch_size)
		optim = self.adamW_optim
		schedule = self.lr_schedule

		training_losses = self.history['train_losses']
		training_costs = self.history['train_costs']
		validation_losses = self.history['val_losses']
		validation_costs = self.history['val_costs']
		learning_rates = self.history['learning_rates']
		if show_plots:
			plt.ion()
			loss_fig, loss_ax = plt.subplots(figsize=(8, 4))
			raw_loss_line, = loss_ax.plot([], [], linestyle='--', color='r', label='training batch loss')
			raw_cost_line = loss_ax.plot([], [], linestyle='--', color='y', label='training batch cost')
			#smoothed_loss_line, = loss_ax.plot([], [], linestyle='-', color='green', label='avg last 50')
			validation_loss_line, = loss_ax.plot([], [], linestyle='-', marker='o', color='purple', label='validation epoch loss')
			loss_ax.set_xlabel('Epoch')
			loss_ax.set_ylabel('NT-Xent loss')
			loss_ax.set_title('Contrastive Training loss')
			loss_ax.legend()
			lr_fig, lr_ax = plt.subplots(figsize=(8, 4))
			lr_line, = lr_ax.plot([], [], linestyle='-', color='blue', label='learning rate')
			lr_ax.set_xlabel('Epoch')
			lr_ax.set_ylabel('Learning rate')
			lr_ax.set_title('Increasing Cyclical Learning Rate')
			lr_ax.legend()
			loss_fig.show()
			lr_fig.show()

		X_train_shuffled = X_train.clone()
		pbar = tqdm.tqdm(range(max_num_epochs))
		update_step = 1
		
		

		original_head = self.layers[-1]
		#pretraining_head = MLP_Layer(input_dim=last_layer_output_shape, output_dim=head_output_dim, activation_function="ReLu")
		#self.layers[-1] = pretraining_head
		try:
			self.__to(self.device)
			self.show_filters(self.layers[0].W, freeze=False)

			for i in pbar:
				if debug_logs: print(f"Started epoch {i}")
				permutation = torch.randperm(X_train.shape[-1])
				if shuffle_set:
					X_train_shuffled = X_train_shuffled[..., permutation]
		
				batch_loss = None
				for j in range(num_minibatches):
					#print(f"minibatch {j}")
					#print(f"Starting update step {update_step}")
					batch_start = j*batch_size
					batch_end = (j+1)*batch_size
					
					X_batch = X_train_shuffled[..., batch_start:batch_end].to(self.device)
					with torch.no_grad():
						aug_batch1 = self.dataloader.augment_data(X_batch, flip_prob=.4, rot_prob=0.1, noise_prob=.4, individual_aug=True)
						aug_batch2 = self.dataloader.augment_data(X_batch, flip_prob=.4, rot_prob=0.1, noise_prob=.4, individual_aug=True)
	
					params = [{"W": layer.W, "b": layer.b} for layer in self.__param_layers]
					if debug_logs: print(f"Magnitude of first W in params before forward: {params[0]['W'].norm()}")
					for param_set in params: # Zeroing gradients before loss calculation
						param_set["W"].grad = None
						param_set["b"].grad = None
					
					output_batch1 = self.forward(aug_batch1, normalize_at=normalize_at) # (10, batch_size)
					output_batch2 = self.forward(aug_batch2, normalize_at=normalize_at) # (10, batch_size)

					if debug_logs: print(f"Output magnitude at step {update_step}: {torch.round(output_batch1.norm(), decimals=2)}")
					loss, cost = self.__get_contrastive_loss(output_batch1, output_batch2, temp=.1) # Choise of temp is based on report
					loss.backward()
					last_filter_norm = self.layers[0].W.norm().item()

					with torch.no_grad():
						auto_grads = [{"W": layer.W.grad, "b": layer.b.grad} for layer in self.__param_layers]

						lr = schedule.get_lr()
						learning_rates.append(lr)
						optim_grads = optim.update_gradient(params=params, grads=auto_grads, iteration=update_step, lam=self.lam) 
						self.__back_pass(lr, optim_grads)
						self.show_filters(self.layers[0].W, freeze=False, step=len(training_losses)/num_minibatches)
						print(f"\n======= Step {update_step} ======= \nLoss: {loss.item():.6f}\nFilter grad: {auto_grads[0]['W'].norm().item():.6f}, diff: {self.layers[0].W.norm().item() - last_filter_norm:.6f}\nOutput magnitude: {output_batch1.norm().item():.6f}\n====================\n")
						batch_loss = loss.detach().cpu().item()
						batch_cost = cost.detach().cpu().item()
						training_losses.append(batch_loss)
						training_costs.append(batch_cost)

						if show_plots:
							epochs = np.arange(1, len(training_losses) + 1)/num_minibatches
							raw_loss_line.set_data(epochs, training_losses)
							raw_cost_line.set_data(epochs, training_costs)
							lr_steps = np.arange(1, len(learning_rates) + 1)/num_minibatches
							lr_line.set_data(lr_steps, learning_rates)
							loss_ax.relim()
							loss_ax.autoscale_view()
							lr_ax.relim()
							lr_ax.autoscale_view()
							loss_fig.canvas.draw()
							loss_fig.canvas.flush_events()
							lr_fig.canvas.draw()
							lr_fig.canvas.flush_events()

					update_step += 1

				with torch.no_grad(): # Checking validation data
					self.current_epoch += 1
					X_val = X_val.to(self.device) # We only do one batch of val

					val_losses = []
					val_costs = []
					num_val_batches = X_val.shape[-1] // batch_size

					for k in range(num_val_batches):
						v_start = k * batch_size
						v_end = (k + 1) * batch_size
						X_val_batch = X_val[..., v_start:v_end].to(self.device)

						val_aug_1 = self.dataloader.augment_data(X_val_batch, flip_prob=0.4, rot_prob=0.1, noise_prob=0.4)
						val_aug_2 = self.dataloader.augment_data(X_val_batch, flip_prob=0.4, rot_prob=0.1, noise_prob=0.4)

						output_batch1 = self.forward(val_aug_1, normalize_at=normalize_at)
						output_batch2 = self.forward(val_aug_2, normalize_at=normalize_at)

						v_loss, v_cost = self.__get_contrastive_loss(output_batch1, output_batch2, temp=0.1)
						val_losses.append(v_loss.detach().cpu().item())
						val_costs.append(v_cost.detach().cpu().item())

					# Mean over all equal-sized validation minibatches
					val_loss_value = float(np.mean(val_losses)) if val_losses else 0.0
					self.history['val_losses'].append(val_loss_value)

					print(f"\n======= Validation ======= \nLoss: {val_loss_value:.6f}\n====================\n")
					if show_plots:
						validation_steps = np.arange(1, len(validation_losses) + 1)
						validation_loss_line.set_data(validation_steps, validation_losses)
						loss_ax.relim()
						loss_ax.autoscale_view()
						loss_fig.canvas.draw()
						loss_fig.canvas.flush_events()

				if save_path is not None and (i % epoch_save_step == 0 or i == max_num_epochs - 1):
					self.save_checkpoint(f"{save_path}_epoch_{self.current_epoch}")

				if training_losses:
					pbar.set_postfix(loss = f"{training_losses[-1]:.4f}", filter_mag = f"{self.layers[0].W.norm().item():.4f}")
		finally:
			self.layers[-1] = original_head # Remove pretraining head
		
		if not show_plots: # We only need the lists below for plots
			return

		plt.ioff()
		loss_fig.canvas.draw()
		lr_fig.canvas.draw()
		plt.show()



	def __get_contrastive_loss(self, aug_set1: torch.Tensor, aug_set2: torch.Tensor, temp: float) -> tuple[torch.Tensor, torch.Tensor]:
		"""Returns the contrastive loss from paper in the report. 
		
		Returns:
			loss (torch.Tensor): The loss. 
			cost (torch.Tensor): The cost. Which is loss + regularisation. """
		
		batch_N = aug_set1.shape[1]
		aug_stack = torch.concat([aug_set1, aug_set2], dim=1)
		aug_stack = torch.nn.functional.normalize(aug_stack, p=2, dim = 0)

		sim_matrix = (aug_stack.T @ aug_stack) / temp
		exp_sim = torch.exp(sim_matrix)
		exp_sim_no_self = exp_sim * (1-torch.eye(2*batch_N).to(self.device)) # Remove self-similarities
		
		pos_1_to_2 = torch.diag(exp_sim_no_self, diagonal=batch_N)
		pos_2_to_1 = torch.diag(exp_sim_no_self, diagonal=-batch_N)
		nominator = torch.cat([pos_1_to_2, pos_2_to_1])
		denominator = torch.sum(exp_sim_no_self, dim=1)

		loss = torch.mean(-torch.log(nominator/denominator) + 1e-8)

		reg = 0
		if self.lam != 0: #Reg is zero if lam is zero
			for layer in self.__param_layers:
				cur_weights:torch.Tensor = layer.W
				reg += self.lam * torch.sum(cur_weights**2)

		return loss, loss + reg
	


	def __similarity(x1: torch.Tensor, x2: torch.Tensor) -> float:
		return (x1.T @ x2) / (torch.norm(x1) * torch.norm(x2))



	def get_accuracy(self, X:torch.Tensor, y:torch.Tensor, batch_size:int=200) -> torch.Tensor:
		"""Get the accuracy of the current network. 
		
		Args:
			X (torch.Tensor): The data. (height, width, nf, n)
			y (torch.Tensor): The correct labels. (n, )
			batch_size (int): Batch size for the forward loop to not get memory error. 
		
		Returns:
			accuracy (torch.Tensor): The accuracy of the predicted labels. (1)
		"""

		n_datapoints = y.shape[0]

		num_minibatches = int(n_datapoints / batch_size)

		total_correct = torch.tensor([0.0]).to(self.device)

		for i in range(num_minibatches):

			batch_start = i*batch_size
			batch_end = (i+1)*batch_size
			X_batch = X[..., batch_start:batch_end]
			X_batch = X_batch.to(self.device)
			y_batch = y[batch_start:batch_end]

			P = self.forward(X_batch, apply_dropout=False) # (10, batch_size)
			guesses = torch.argmax(P, axis = 0).float()

			corrects = torch.isclose(guesses, y_batch).float()

			total_correct += torch.sum(corrects)

		return total_correct / n_datapoints

	
	def display_accuracy_histogram(self, X, y):
		P = self.forward(X) # (10, batch_size)
		guesses = torch.argmax(P, axis = 0)
		corrects = torch.isclose(guesses, y)

		#print(corrects.shape)
		#print(corrects)

		class_corrects = torch.array(y)[corrects]
		plt.hist(class_corrects)
		plt.show()

		return torch.sum(corrects) / X.shape[1]


	def save_checkpoint(self, filepath: str, include_training_state: bool = True):
		"""Saves the model weights to a file.

		If include_training_state is true, the training history, epoch counter, optimizer state,
		and learning-rate schedule state are stored as well.
		"""
		model_state = []
		for layer in self.__param_layers:
			layer_state = {
				"W": layer.W.detach().cpu().clone(),
				"b": layer.b.detach().cpu().clone()
			}
			model_state.append(layer_state)

		checkpoint = {"model_state": model_state}
		if include_training_state:
			optim_state = self.adamW_optim.get_state() if self.adamW_optim is not None else None
			lr_schedule_state = self.lr_schedule.get_state() if self.lr_schedule is not None and hasattr(self.lr_schedule, "get_state") else None
			checkpoint.update({
				"history": self.history,
				"current_epoch": self.current_epoch,
				"optim_state": optim_state,
				"lr_schedule_state": lr_schedule_state
			})
		torch.save(checkpoint, "Saved_Networks/" + filepath)
		print(f"Checkpoint successfully saved to {filepath}")


	def load_checkpoint(self, filepath: str, load_training_state: bool = True, reinitialize_last: int = 0):
		"""Loads a checkpoint.

		If load_training_state is false, only the model weights are restored.
		If load_last_layer is false, the final trainable layer is left untouched.
		"""
		try:
			checkpoint = torch.load("Saved_Networks/" + filepath)
		except:
			print("No saved file found, returning...")
			return
		
		# Check if it's the old format (just a list of layers) or the new dictionary format
		if isinstance(checkpoint, list): 
			model_state = checkpoint
		else:
			model_state = checkpoint.get("model_state")
			if load_training_state:
				self.history = checkpoint.get("history", self.history)
				self.total_training_time = checkpoint.get("total_training_time", 0.0)
				self.current_epoch = checkpoint.get("current_epoch", 0)

				self.adamW_optim = AdamW()
				self.adamW_optim.set_state(checkpoint.get("optim_state", None))

				self.lr_schedule = cyclic_increasing_lr()
				self.lr_schedule.set_state(checkpoint.get("lr_schedule_state", None))
			else:
				self.history = {
					'epochs': [], 'train_steps': [], 'train_costs': [], 'train_losses': [],
					'val_steps': [], 'val_costs': [], 'val_losses': [],
					'train_accuracys': [], 'val_accuracys': [], 'learning_rates': []
				}
				self.total_training_time = 0.0
				self.current_epoch = 0

			

		layers_to_load = self.__param_layers if reinitialize_last <= 0 else self.__param_layers[:-reinitialize_last]
		model_state_to_load = model_state if reinitialize_last <= 0 else model_state[:len(layers_to_load)]

		if len(model_state_to_load) != len(layers_to_load):
			raise ValueError("The loaded model has a different number of layers than the current model!")
			
		with torch.no_grad():
			for layer, loaded_state in zip(layers_to_load, model_state_to_load):
				layer.W.copy_(loaded_state["W"])
				layer.b.copy_(loaded_state["b"])

		for layer in self.__param_layers:
			layer.W.requires_grad = True
			layer.b.requires_grad = True
		print(f"Checkpoint successfully loaded from {filepath}")

	
	def show_filters(self, F: torch.Tensor, freeze: bool = False, epoch = 0):
		# F (the first conv layer) will have shape (nf, 3, width of filters, height of filters)
		
		nf = F.shape[0]
		num_to_show = min(64, nf)

		if not hasattr(self, 'filter_fig') or not plt.fignum_exists(self.filter_fig.number):
			plt.ion() 
			ncols = math.ceil(math.sqrt(num_to_show))
			nrows = math.ceil(num_to_show / ncols)
			self.filter_fig, self.filter_axs = plt.subplots(nrows, ncols, figsize=(10, 5))
			plt.show(block=False)
			self.filter_axs = self.filter_axs.flatten()
			
			# Turn off any unused axes (grid may have more cells than num_to_show)
			for j in range(num_to_show, len(self.filter_axs)):
				self.filter_axs[j].axis('off')

			self.im_objects = [None] * len(self.filter_axs)

		if not plt.fignum_exists(self.filter_fig.number):
			return

		F_visual = F.detach().clone()
		max_abs = torch.max(torch.abs(F_visual)).item()
		F_visual = F_visual.permute((2, 3, 1, 0)).contiguous()

		for i in range(num_to_show):
			im = (F_visual[..., i] / (2 * max_abs + 1e-8)) + 0.5
			im = torch.clamp(im, 0.0, 1.0).cpu().numpy()
			
			# 2. Update pixels instead of redrawing the whole plot!
			if self.im_objects[i] is None:
				# First time: draw the image and save the object
				self.im_objects[i] = self.filter_axs[i].imshow(im)
				self.filter_axs[i].axis('off')
			else:
				# Next epochs: just inject the new pixels! (Lightning fast)
				self.im_objects[i].set_data(im)
		
		if freeze: 
			plt.ioff()
			plt.show()
		else:
			self.filter_fig.suptitle(f"Filters at epoch: {epoch}")
			self.filter_fig.canvas.draw()
			self.filter_fig.canvas.flush_events()

if __name__ == "__main__":
	pass
