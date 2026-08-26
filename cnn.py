import matplotlib.pyplot as plt
import torch
import tqdm
from dataloader import Dataloader
from convolution import Conv_Layer
from pooling import Pool_Layer
from mlp import MLP_Layer
from utils import *
import numpy as np

class CNN:
	def __init__(self, layers: list[Conv_Layer | MLP_Layer], lam: float = 0, dropout_prob:float=0, seed:int=42):
		"""Creates a linear classifier with 10 output classes.
		Args:
			layers (list[Conv_Layer | MLP_Layer]): A list of either convolution-layers or mlp-layers, already initialized.
			lam (float): The lambda used in L2 regularization of the layers during backward pass. 
			dropout_prob (float): Probability of a node being turned off for each update step. 
			seed (int): The seed for the random number generator. """
		self.layers = layers
		"""All the neural network layers. """
		self.lam = lam
		"""The regularisation term. """
		self.dropout_prob = dropout_prob
		"""Probability of a node being turned off for each update step. """
		self.dataloader = Dataloader()
		"""The dataloader class for this CNN. """

		self.history = {
			'epochs': [], 'train_costs': [], 'train_losses': [],
			'val_costs': [], 'val_losses': [],
			'train_accuracys': [], 'val_accuracys': []
		}
		"""The history of the training run. \\
			'epochs', 'train_costs', 'train_losses', 'val_costs', 'val_losses',
			'train_accuracys', 'val_accuracys'"""
	
		self.current_epoch = 0
		"""Current epoch. """


	def forward(self, x: torch.Tensor, dropout_prob:float) -> torch.Tensor:
		"""Do a forward pass of the classifier with the loaded Mx (needs to be loaded!)
		
		Args:
			x (torch.Tensor): The input of the whole network. (height, width, nf, n)
			dropout_prob (float): Probability of an output node being turned off for the forward pass.
			
		Returns:
			x (torch.Tensor): The output of the whole network. (output_dim, n)"""

		prev_layer = None
		for layer in self.layers:

			if prev_layer != None and not isinstance(prev_layer, MLP_Layer) and isinstance(layer, MLP_Layer): # If MLP and prev layer is not MLP

				#Flatten input
				x = x.reshape(x.shape[0]*x.shape[1]*x.shape[2], x.shape[3])

			x = layer.forward(x, dropout_prob)
			if torch.any(x.isnan()):
				raise ValueError("One item is NaN. ")
			prev_layer = layer
			
		return x


	def __cost_function(self, p: torch.Tensor, Y_batch: torch.Tensor):
		"""Calculated the mean cross-entropy loss of a set of 
		predicted class probabilites"""
		# both have dimensions (10, batch_size)
		# We take the mean of the loss over the batch
		cross_entropy = self.__cross_entropy_loss(p, Y_batch)
		
		reg = 0
		for layer in self.__param_layers:
			cur_weights = layer.W
			reg += self.lam * torch.sum(cur_weights**2)

		return cross_entropy + reg, cross_entropy


	def __cross_entropy_loss(self, p: torch.Tensor, Y_batch: torch.Tensor, eps:float=1e-10) -> float:
		"""Calculates the cross entropy loss. 
		
		Args:
			p (torch.Tensor): The calculated probabilites. (n_classes, n)
			Y_batch (torch.Tensor): The actual one-hot labels. (n_classes, n)
			eps (float): Small number to avoid overflow errors when doing log(p). """

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


	def minibatchGD(self, X_train:torch.Tensor, Y_train:torch.Tensor, y_train:torch.Tensor, X_val:torch.Tensor, Y_val:torch.Tensor, 
				 y_val:torch.Tensor, schedule: lr_schedule, optim: optimizer, max_num_epochs: int = 40, batch_size: int = 100, 
					shuffle_set: bool = True, show_plots: bool = True, epoch_plot_period: int = 10):
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
			show_plots (bool): If plots of the training and validation loss, cost and accuracy should be displayed at the end. """

		n_datapoints = X_train.shape[3]
		num_minibatches = int(n_datapoints / batch_size)

		val_cost = torch.inf
		
		if show_plots:
			plt.ion() # Turn on interactive mode
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

		pbar = tqdm.tqdm(range(max_num_epochs))
		update_step = 1

		for i in pbar:
			#print(f"Started epoch {i}")
			self.current_epoch += 1

			permutation = torch.randperm(n_datapoints)
			if shuffle_set:
				X_train_shuffled = X_train[..., permutation]
				Y_train_shuffled = Y_train[:, permutation]
	
			
			total_train_cost = 0
			total_train_loss = 0
			for j in range(num_minibatches):
				#print(f"minibatch {j}")
				batch_start = j*batch_size
				batch_end = (j+1)*batch_size

				X_batch = X_train_shuffled[..., batch_start:batch_end]
				with torch.no_grad():
					#X_aug = self.dataloader.augment_data(X=X_batch)
					X_aug = X_batch
					
				P_batch = self.forward(X_aug, self.dropout_prob)
				Y_batch = Y_train_shuffled[:, batch_start:batch_end]
				batch_cost, batch_loss = self.__cost_function(P_batch, Y_batch)

				total_train_cost += batch_cost.detach()
				total_train_loss += batch_loss.detach()

				lr = schedule.get_lr(update_step)
				params:dict[str, torch.Tensor] = [{"W": layer.W, "b": layer.b} for layer in self.__param_layers]

				for param_set in params: # Zeroing gradients before loss calculation
					param_set["W"].grad = None
					param_set["b"].grad = None

				batch_cost.backward()
				with torch.no_grad():
					auto_grads:dict[str, float] = [{"W": layer.W.grad, "b": layer.b.grad} for layer in self.__param_layers]
					optim_grads = optim.update_gradient(params=params, grads=auto_grads, iteration=update_step, lam=self.lam) 
					lr = schedule.get_lr(update_step)
					self.__back_pass(lr, optim_grads)

				# REMOVED OUR OWN GRADIENT CALCULATION
				#grads = self.get_gradient(Mx_batch, Y_batch, P_batch)
				#optim_grads = optim.update_gradient(params=params, grads=grads, iteration=update_step, lam=self.lam) 
				#self.__back_pass(lr, optim_grads)
				
				update_step += 1
			

			if not show_plots or (i%epoch_plot_period != 0 and i != max_num_epochs-1): # We only need the lists below for plots
				continue

			with torch.no_grad():
				self.history['epochs'].append(self.current_epoch)
				self.history['train_costs'].append((total_train_cost.detach()/num_minibatches).item())
				self.history['train_losses'].append((total_train_loss.detach()/num_minibatches).item())
				
				P_val = self.forward(X_val, dropout_prob=0)
				val_cost, val_loss = self.__cost_function(P_val, Y_val)
				self.history['val_costs'].append(val_cost.detach().item())
				self.history['val_losses'].append(val_loss.detach().item())
			
				if X_val is not None and y_val is not None and y_train is not None:
					self.history['val_accuracys'].append(self.get_accuracy(X_val, y_val).item())
					self.history['train_accuracys'].append(self.get_accuracy(X_train, y_train).item())

			pbar.set_postfix(loss = f"{round(val_loss.item(), 4)}")
			
			nbr_entries = len(self.history['train_costs'])
			x_axis = self.history['epochs']
			ax1.clear()
			ax1.plot(x_axis, self.history['train_costs'], label = "Training Cost", color = "y")
			ax1.plot(x_axis, self.history['train_losses'], label = "Training Loss", linestyle='--', color = "r")
			ax1.plot(x_axis, self.history['val_costs'], label = "Validation Cost", color = "b")
			ax1.plot(x_axis, self.history['val_losses'], label = "Validation Loss", linestyle = '--', color = "c")
			ax1.set_xlabel("Epoch")
			ax1.legend()
			ax1.set_title(f"Train/val cost + loss, lam={self.lam}")

			ax2.clear()
			ax2.set_title("Train/Val accuracy over epochs")
			ax2.plot(x_axis, self.history['val_accuracys'], label="validation")
			ax2.plot(x_axis, self.history['train_accuracys'], label="training")
			ax2.set_xlabel("Epoch")
			ax2.legend()

			plt.pause(0.001)

		if show_plots:
			plt.ioff() # Turn off interactive mode
			plt.show() # Keep the final plot window open when training finishes

	def contrastiveLearning(self, X_train: torch.Tensor, schedule: lr_schedule, optim: optimizer,
					 max_num_epochs: int = 40, batch_size: int = 100, 
					shuffle_set: bool = True, show_plots: bool = True, debug_logs: bool = False):
		""" Does contrastive self-supervised learning """

		n_datapoints = X_train.shape[-1]
		num_minibatches = int(n_datapoints / batch_size)

		training_losses = []

		X_train_shuffled = X_train.clone()
		pbar = tqdm.tqdm(range(max_num_epochs))
		update_step = 1
		
		
		# Adding pretraining head
		last_layer_output_shape = self.layers[-1].W.shape[1]
		head_output_dim = 200
		original_head = self.layers[-1]
		pretraining_head = MLP_Layer(input_dim=last_layer_output_shape, output_dim=head_output_dim, activation_function="linear")
		try:
			self.layers[-1] = pretraining_head

			for i in pbar:
				if debug_logs: print(f"Started epoch {i}")

				permutation = torch.randperm(X_train.shape[-1])
				if shuffle_set:
					X_train_shuffled = X_train_shuffled[..., permutation]
		
				epoch_loss = None
				for j in range(num_minibatches):
					#print(f"minibatch {j}")
					#print(f"Starting update step {update_step}")
					batch_start = j*batch_size
					batch_end = (j+1)*batch_size
					
					X_batch = X_train_shuffled[..., batch_start:batch_end]
					aug_batch1 = self.dataloader.augment_data(X_batch, flip_prob=.5, rot_prob=0, noise_prob=.5)
					aug_batch2 = self.dataloader.augment_data(X_batch, flip_prob=.5, rot_prob=0, noise_prob=1)
					aug_batch1 = self.dataloader.create_Mx(aug_batch1, filter_width=self.layers[0].filter_width)
					aug_batch2 = self.dataloader.create_Mx(aug_batch2, filter_width=self.layers[0].filter_width)
					#print(f"Mag of augmented batch 1 Mx: {aug_batch1.norm()}")
					
					params = [{"W": layer.W, "b": layer.b} for layer in self.layers]
					if debug_logs: print(f"Magnitude of first W in params before forward: {params[0]['W'].norm()}")
					for param_set in params: # Zeroing gradients before loss calculation
						param_set["W"].grad = None
						param_set["b"].grad = None
					
					output_batch1 = self.forward(aug_batch1, dropout_prob=.3) # (10, batch_size)
					output_batch2 = self.forward(aug_batch2, dropout_prob=.3) # (10, batch_size)
					if debug_logs: print(f"Output magnitude at step {update_step}: {torch.round(output_batch1.norm(), decimals=2)}")
					loss = self.__get_contrastive_loss(output_batch1, output_batch2, temp=0.1) # Choise of temp is based on report
					#print(f"Loss at step {update_step}: {torch.round(loss, decimals=2)}")
					loss.backward()
					with torch.no_grad():
						auto_grads = [{"W": layer.W.grad, "b": layer.b.grad} for layer in self.layers]
						#print(f"Grad of input layer at step {update_step}: {torch.round(auto_grads[0]["W"].norm(), decimals=2)}")
						#print(f"Grad of output layer at step {update_step}: {torch.round(auto_grads[-1]["W"].norm(), decimals=2)}")
						optim_grads = optim.update_gradient(params=params, grads=auto_grads, iteration=update_step, lam=self.lam) 
						lr = schedule.get_lr(update_step)
						self.__back_pass(lr, optim_grads)

					epoch_loss = loss.detach().numpy()
					update_step += 1
				
				self.show_filters(self.layers[0].W, freeze=False)
				training_losses.append(epoch_loss)
				pbar.set_postfix(loss = f"{training_losses[-1]:.4f}", filter_mag = f"{self.layers[0].W.norm().item():.4f}")
		finally:
			self.layers[-1] = original_head # Remove pretraining head
		
		if not show_plots: # We only need the lists below for plots
			return


		plt.plot(np.arange(len(training_losses))*num_minibatches, training_losses, linestyle='--', color = "r")
		plt.xlabel("Update step")
		plt.title("Contrastive Training loss")
		plt.show()



	def __get_contrastive_loss(self, aug_set1: torch.Tensor, aug_set2: torch.Tensor, temp: float) -> torch.Tensor:
		"""Returns the contrastive loss from paper in the report. """
		batch_N = aug_set1.shape[1]
		aug_stack = torch.concat([aug_set1, aug_set2], dim=1)
		aug_stack = torch.nn.functional.normalize(aug_stack, p=2, dim = 0)

		sim_matrix = (aug_stack.T @ aug_stack) / temp
		exp_sim = torch.exp(sim_matrix)
		exp_sim_no_self = exp_sim * (1-torch.eye(2*batch_N)) # Remove self-similarities
		
		positive_sims = torch.diag(exp_sim_no_self, diagonal=batch_N)
		nominator = torch.cat([positive_sims, positive_sims])
		denominator = torch.sum(exp_sim_no_self, dim=1)

		return torch.mean(-torch.log(nominator/denominator) + 1e-8)
	


	def __similarity(x1: torch.Tensor, x2: torch.Tensor) -> float:
		return (x1.T @ x2) / (torch.norm(x1) * torch.norm(x2))



	def get_accuracy(self, X, y):
		P = self.forward(X, dropout_prob=0) # (10, batch_size)
		guesses = torch.argmax(P, axis = 0).float()

		corrects = torch.isclose(guesses, y).float()

		return torch.sum(corrects) / P.shape[1]

	
	def display_accuracy_histogram(self, X, y):
		P = self.forward(X, dropout_prob=0) # (10, batch_size)
		guesses = torch.argmax(P, axis = 0)
		corrects = torch.isclose(guesses, y)

		#print(corrects.shape)
		#print(corrects)

		class_corrects = torch.array(y)[corrects]
		plt.hist(class_corrects)
		plt.show()

		return torch.sum(corrects) / X.shape[1]


	def save_checkpoint(self, filepath: str):
		"""Saves the weights, biases, history, and elapsed time to a file."""
		model_state = []
		for layer in self.__param_layers:
			layer_state = {
				"W": layer.W.detach().cpu().clone(),
				"b": layer.b.detach().cpu().clone()
			}
			model_state.append(layer_state)
		
		checkpoint = {
			"model_state": model_state,
			"history": self.history,
			"current_epoch": self.current_epoch
		}
		torch.save(checkpoint, "Saved_Networks/" + filepath)
		print(f"Checkpoint successfully saved to {filepath}")


	def load_checkpoint(self, filepath: str):
		"""Loads a checkpoint so you can resume training and plots."""
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
			self.history = checkpoint.get("history", self.history)
			self.total_training_time = checkpoint.get("total_training_time", 0.0)
			self.current_epoch = checkpoint.get("current_epoch", 0)

		if len(model_state) != len(self.__param_layers):
			raise ValueError("The loaded model has a different number of layers than the current model!")
			
		with torch.no_grad():
			for layer, loaded_state in zip(self.__param_layers, model_state):
				layer.W.copy_(loaded_state["W"])
				layer.W.requires_grad = True
				layer.b.copy_(loaded_state["b"])
				layer.b.requires_grad = True
				
		print(f"Checkpoint successfully loaded from {filepath}")

	
	def show_filters(self, F: torch.Tensor, freeze: bool = False):
		# 1. Initialize the figure and image objects ONLY ONCE
		if not hasattr(self, 'filter_fig') or not plt.fignum_exists(self.filter_fig.number):
			plt.ion() 
			self.filter_fig, self.filter_axs = plt.subplots(8, 8, figsize=(10, 5))
			plt.show(block=False)
			self.filter_axs = self.filter_axs.flatten()
			
			# Create a list to store the actual Image objects
			self.im_objects = [None] * len(self.filter_axs)

		if not plt.fignum_exists(self.filter_fig.number):
			return

		F_visual = F.detach().clone()
		nf = F.shape[1]
		f = int(torch.math.sqrt(F.shape[0]/3))
		max_abs = torch.max(torch.abs(F_visual)).item()
		F_visual = F_visual.reshape((f, f, 3, nf))

		num_to_show = min(64, nf)
		
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
			self.filter_fig.canvas.draw()
			self.filter_fig.canvas.flush_events()

if __name__ == "__main__":
	pass
