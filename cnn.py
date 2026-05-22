import matplotlib.pyplot as plt
import torch
import tqdm
from dataloader import Dataloader
from convolution import Conv_Layer
from mlp import MLP_Layer
from utils import *

class CNN:
	def __init__(self, layers: list[Conv_Layer | MLP_Layer], lam: float = 0, seed:int=42):
		"""Creates a linear classifier with 10 output classes.
		Args:
			layers: A list of either convolution-layers or mlp-layers, already initialized.
			lam: The lambda used in L2 regularization of the layers during backward pass"""
		self.layers = layers
		self.lam = lam
		self.dataloader = Dataloader()
		self.rng = self.__initialize_seed(seed)
	
	def __initialize_seed(self, seed:int) -> torch.Generator:
		"""Initialize the random number generator. 
		
		Args:
			seed (int): The seed. 
		
		Returns:
			rng (Generator): The random number generator. """
		
		rng = torch.Generator()
		
		rng.manual_seed(seed)
	
		return rng


	def forward(self, Mx: torch.Tensor = None):
		"""Do a forward pass of the classifier with the loaded Mx (needs to be loaded!)"""
		x = Mx
		for layer in self.layers:
			x = layer.forward(x)
		return x


	def __cost_function(self, p: torch.Tensor, Y_batch: torch.Tensor):
		"""Calculated the mean cross-entropy loss of a set of 
		predicted class probabilites"""
		# both have dimensions (10, batch_size)
		# We take the mean of the loss over the batch
		cross_entropy = self.__cross_entropy_loss(p, Y_batch)
		
		reg = 0
		for layer in self.layers:
			cur_weights = layer.W
			reg += self.lam * torch.sum(cur_weights**2)

		return cross_entropy + reg


	def __cross_entropy_loss(self, p: torch.Tensor, Y_batch: torch.Tensor):
		return torch.mean(torch.sum(-Y_batch*torch.log(p), axis = 0))


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
				* **'W'** (torch.Tensor): Gradient of the weights in the MLP. (local_output_dim, local_input_dim)
				* **'F'** (torch.Tensor): Gradient of the weights in the convolution. (local_output_dim, local_input_dim)
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
				g_batch = g_batch * (layer.h > 0).T

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


	def __back_pass(self, Mx_batch:torch.Tensor, Y_batch:torch.Tensor, P_batch:torch.Tensor,
				 lr: float, grads: list[dict[str, float]]) -> None:
		"""Updates the network based on the gradients. 
		
		Args:
			Mx_batch (torch.Tensor): The input data batch which was used for the current training batch. (n_filters, filter_size, batch_size)
			Y_batch (torch.Tensor): The output/class data batch which was used for the current training batch. (num_labels, batch_size)
			P_batch (torch.Tensor): The prediction probabilities for the current batch. (num_labels, batch_size),
			lr: learning rate,
			grads: list of dictionaries containign gradient for each parameter.
			"""

		for i in range(len(grads)):
			cur_layer = self.layers[i]
			cur_layer.W -= lr * grads[i]['W']
			cur_layer.b -= grads[i]['b']


	def minibatchGD(self, Mx_train, Y_train, y_train, Mx_val, Y_val, y_val, schedule: lr_schedule, optim: optimizer,
					 max_num_epochs: int = 40, batch_size: int = 100, 
					shuffle_set: bool = True, show_plots: bool = True):
		""" Does end-to-end minibatch gradient descent with the given training data. Has options for """

		n_datapoints = Mx_train.shape[2]
		num_minibatches = int(n_datapoints / batch_size)

		train_costs = []
		train_losses = []
		val_costs = []
		val_losses = []
		lr_decay_epochs = []
		val_accuracys = []
		train_accuracys = []

		Mx_train_shuffled = Mx_train.clone()
		Y_train_shuffled = Y_train.clone()
		val_cost = torch.inf
		
		pbar = tqdm.tqdm(range(max_num_epochs))
		update_step = 1

		for i in pbar:
			#print(f"Started epoch {i}")

			permutation = torch.randperm(n_datapoints, generator=self.rng)
			if shuffle_set:
				Mx_train_shuffled = Mx_train_shuffled[:, :, permutation]
				Y_train_shuffled = Y_train_shuffled[:, permutation]
	
			
			for j in range(num_minibatches):
				#print(f"minibatch {j}")
				batch_start = j*batch_size
				batch_end = (j+1)*batch_size

				Mx_batch = Mx_train_shuffled[:, :, batch_start:batch_end]
				P_batch = self.forward(Mx_batch)
				Y_batch = Y_train_shuffled[:, batch_start:batch_end]

				lr = schedule.get_lr(update_step)
				params = [{"W": layer.W, "b": layer.b} for layer in self.layers]
				grads = self.get_gradient(Mx_batch, Y_batch, P_batch)
				optim_grads = optim.update_gradient(params=params, grads=grads, iteration=update_step, lam=self.lam) 
				
				self.__back_pass(Mx_batch, Y_batch, P_batch, lr, optim_grads)
				update_step += 1
			

			if not show_plots: # We only need the lists below for plots
				continue

			P_train = self.forward(Mx_train)
			train_cost = self.__cost_function(P_train, Y_train)
			train_loss = self.__cross_entropy_loss(P_train, Y_train)
			train_costs.append(train_cost)
			train_losses.append(train_loss)
			
			P_val = self.forward(Mx_val)
			val_cost = self.__cost_function(P_val, Y_val)
			val_loss = self.__cross_entropy_loss(P_val, Y_val)
			val_costs.append(val_cost)
			val_losses.append(val_loss)
			
			if Mx_val is not None and y_val is not None and y_train is not None:
				val_accuracys.append(self.get_accuracy(Mx_val, y_val))
				train_accuracys.append(self.get_accuracy(Mx_train, y_train))

			pbar.set_postfix(loss = f"{round(val_loss.item(), 4)}")

		if not show_plots: 
			return
		
		plt.plot(torch.arange(i+1)*num_minibatches, train_costs, label = "Training Cost", color = "y")
		plt.plot(torch.arange(i+1)*num_minibatches, train_losses, label = "Training Loss", linestyle='--', color = "r")
		plt.vlines(lr_decay_epochs, ymin = 1.5, ymax = 2, label = "Lr decay epochs")
		plt.plot(torch.arange(i+1)*num_minibatches, val_costs, label = "Validation Cost", color = "b")
		plt.plot(torch.arange(i+1)*num_minibatches, val_losses, label = "Validation Loss", linestyle = '--', color = "c")
		plt.xlabel("Update step")
		plt.legend()
		plt.title(f"Train/val cost + loss, lam={self.lam}")
		plt.show()

		plt.title("Train/Val accuracy over epochs")
		plt.plot(torch.arange(i+1)*num_minibatches, val_accuracys, label="validation")
		plt.plot(torch.arange(i+1)*num_minibatches, train_accuracys, label="training")
		plt.xlabel("Update step")
		plt.legend()
		plt.show()

	
	def get_accuracy(self, X, y):
		P = self.forward(X) # (10, batch_size)
		guesses = torch.argmax(P, axis = 0).float()

		corrects = torch.isclose(guesses, y).float()

		return torch.sum(corrects) / P.shape[1]
	
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


if __name__ == "__main__":
	pass
