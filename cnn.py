import matplotlib.pyplot as plt
import torch
import tqdm
from dataloader import Dataloader
from convolution import Conv_Layer
from mlp import MLP_Layer
from utils import *

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
		self.rng = self.__initialize_seed(seed)
		"""The random number generator for this CNN. """
	
	def __initialize_seed(self, seed:int) -> torch.Generator:
		"""Initialize the random number generator. 
		
		Args:
			seed (int): The seed. 
		
		Returns:
			rng (Generator): The random number generator. """
		
		rng = torch.Generator()
		
		rng.manual_seed(seed)
	
		return rng


	def forward(self, Mx: torch.Tensor, dropout_prob:float):
		"""Do a forward pass of the classifier with the loaded Mx (needs to be loaded!)"""
		x = Mx
		for layer in self.layers:
			x = layer.forward(x, dropout_prob)
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
			cur_layer = self.layers[i]
			with torch.no_grad():
				cur_layer.W -= lr * grads[i]['W']
				cur_layer.b -= lr * grads[i]['b']


	def minibatchGD(self, X_train:torch.Tensor, Y_train:torch.Tensor, y_train:torch.Tensor, X_val:torch.Tensor, Y_val:torch.Tensor, 
				 y_val:torch.Tensor, schedule: lr_schedule, optim: optimizer, max_num_epochs: int = 40, batch_size: int = 100, 
					shuffle_set: bool = True, show_plots: bool = True):
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

		train_costs = []
		train_losses = []
		val_costs = []
		val_losses = []
		lr_decay_epochs = []
		val_accuracys = []
		train_accuracys = []

		X_train_shuffled = X_train.clone()
		Y_train_shuffled = Y_train.clone()
		Mx_train = self.dataloader.create_Mx(X_train, filter_width=self.layers[0].filter_width)
		Mx_val = self.dataloader.create_Mx(X_val, filter_width=self.layers[0].filter_width)
		val_cost = torch.inf
		
		pbar = tqdm.tqdm(range(max_num_epochs))
		update_step = 1

		for i in pbar:
			#print(f"Started epoch {i}")

			permutation = torch.randperm(n_datapoints, generator=self.rng)
			if shuffle_set:
				X_train_shuffled = X_train_shuffled[..., permutation]
				Y_train_shuffled = Y_train_shuffled[:, permutation]
	
			
			for j in range(num_minibatches):
				#print(f"minibatch {j}")
				batch_start = j*batch_size
				batch_end = (j+1)*batch_size

				X_batch = X_train_shuffled[..., batch_start:batch_end]
				Mx_batch = self.dataloader.create_Mx(X=X_batch, filter_width=self.layers[0].filter_width)
				P_batch = self.forward(Mx_batch, self.dropout_prob)
				Y_batch = Y_train_shuffled[:, batch_start:batch_end]

				lr = schedule.get_lr(update_step)
				params = [{"W": layer.W, "b": layer.b} for layer in self.layers]
				grads = self.get_gradient(Mx_batch, Y_batch, P_batch, )
				optim_grads = optim.update_gradient(params=params, grads=grads, iteration=update_step, lam=self.lam) 
				
				self.__back_pass(lr, optim_grads)
				update_step += 1
			

			if not show_plots: # We only need the lists below for plots
				continue

			P_train = self.forward(Mx_train, dropout_prob=0)
			train_cost = self.__cost_function(P_train, Y_train)
			train_loss = self.__cross_entropy_loss(P_train, Y_train)
			train_costs.append(train_cost.detach())
			train_losses.append(train_loss.detach())
			
			P_val = self.forward(Mx_val, dropout_prob=0)
			val_cost = self.__cost_function(P_val, Y_val)
			val_loss = self.__cross_entropy_loss(P_val, Y_val)
			val_costs.append(val_cost.detach())
			val_losses.append(val_loss.detach())
			
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

	def contrastiveLearning(self, X_train: torch.Tensor, schedule: lr_schedule, optim: optimizer,
					 max_num_epochs: int = 40, batch_size: int = 100, 
					shuffle_set: bool = True, show_plots: bool = True):
		""" Does contrastive self-supervised learning """

		n_datapoints = X_train.shape[1]
		num_minibatches = int(n_datapoints / batch_size)

		training_losses = []

		X_train_shuffled = X_train.clone()
		pbar = tqdm.tqdm(range(max_num_epochs))
		update_step = 1

		# We need a seperate head for the contrastive training
		pretraining_layers = self.layers.copy()
		normal_output_shape = pretraining_layers[-1].W.shape
		pretraining_layers[-1] = MLP_Layer(input_dim=normal_output_shape[1], output_dim=normal_output_shape[0], activation_function="ReLu")

		for i in pbar:
			#print(f"Started epoch {i}")

			permutation = torch.randperm(n_datapoints, generator=self.rng)
			if shuffle_set:
				X_train_shuffled = X_train_shuffled[:, :, permutation]
	
			epoch_loss = None
			for j in range(num_minibatches):
				#print(f"minibatch {j}")
				
				batch_start = j*batch_size
				batch_end = (j+1)*batch_size
				
				X_batch = X_train_shuffled[:, :, batch_start:batch_end]
				aug_batch1 = self.dataloader.augment_data(X_batch)
				aug_batch2 = self.dataloader.augment_data(X_batch)
				
				#aug_batch1 = self.dataloader.create_Mx(aug_batch1, filter_width=self.layers[0].filter_width)
				#aug_batch2 = self.dataloader.create_Mx(aug_batch2, filter_width=self.layers[0].filter_width)

				params = [{"W": layer.W, "b": layer.b} for layer in pretraining_layers]
				for param_set in params: # Zeroing gradients before loss calculation
					param_set["W"].grad = None
					param_set["b"].grad = None
				
				aug_batch1 = self.forward(aug_batch1, dropout_prob=.3) # (10, batch_size)
				aug_batch2 = self.forward(aug_batch2, dropout_prob=.3) # (10, batch_size)
				
				loss = self.__get_contrastive_loss(aug_batch1, aug_batch2, temp=0.1) # Choise of temp is based on report
				loss.backward()
				auto_grads = [{"W": layer.W.grad, "b": layer.b.grad} for layer in pretraining_layers]
				
				optim_grads = optim.update_gradient(params=params, grads=auto_grads, iteration=update_step, lam=self.lam) 
				lr = schedule.get_lr(update_step)
				self.__back_pass(lr, optim_grads)

				epoch_loss = loss
				update_step += 1
			
			training_losses.append(epoch_loss)
			pbar.set_postfix(loss = f"Contrastive training loss: {round(training_losses[-1], 4)}")
			

		if not show_plots: # We only need the lists below for plots
			return


		plt.plot(torch.arange(len(training_losses))*num_minibatches, training_losses, linestyle='--', color = "r")
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
		exp_sim *= (1-torch.eye(2*batch_N)) # Remove self-similarities
		
		positive_sims = torch.diag(exp_sim, diagonal=batch_N)
		nominator = torch.cat([positive_sims, positive_sims])
		denominator = torch.sum(exp_sim, dim=1)

		return torch.mean(-torch.log(nominator/denominator))
	
	def __similarity(x1: torch.Tensor, x2: torch.Tensor) -> float:
		return (x1.T @ x2) / (torch.norm(x1) * torch.norm(x2))



	def __get_contrastive_loss_for(self, aug_set1: torch.Tensor, aug_set2: torch.Tensor, temp: float) -> torch.Tensor:
		"""Returns the contrastive loss from paper in the report. """
		loss_ij = torch.tensor(0)
		loss_ji = torch.tensor(0)
		N_batch = aug_set1.shape[1]

		for img_idx in range(N_batch):
			aug_img1 = aug_set1[:, img_idx]
			aug_img2 = aug_set2[:, img_idx]

			# Normalized Temperature scaled Cross Entropy
			nom_ij= torch.exp(self.__similarity(aug_img1, aug_img2)/temp)
			denom_ij = 0
			nom_ji= torch.exp(self.__similarity(aug_img2, aug_img1)/temp)
			denom_ji = 0
			for i in range(aug_set1.shape[1]):
				if i == img_idx: 
					continue
				neg1 = aug_set1[:, i]
				neg2 = aug_set2[:, i]
				denom_ij += torch.exp(self.__similarity(aug_img1, neg1)/temp)
				denom_ij += torch.exp(self.__similarity(aug_img1, neg2)/temp)

				denom_ji += torch.exp(self.__similarity(aug_img2, neg1)/temp)
				denom_ji += torch.exp(self.__similarity(aug_img2, neg2)/temp)
			
			loss_ij += -torch.log(nom_ij/denom_ij)
			loss_ji += -torch.log(nom_ji/denom_ji)
		
		return (loss_ij + loss_ji) / (2*N_batch)







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


if __name__ == "__main__":
	pass
