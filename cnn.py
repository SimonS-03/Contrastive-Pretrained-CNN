import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import tqdm
from convolution import Convolution_Layer

class CNN:
    def __init__(self, layers: list[Convolution_Layer], lam: float = 0):
        """Creates a linear classifier with 10 output classes.
        Args:
            layers: A list of either convolution-layers or mlp-layers, already initialized.
            lam: The lambda used in L2 regularization of the layers during backward pass"""
        self.layers = layers
        self.lam = lam

    
    def __get_cyclic_lr(self, min_lr, max_lr, cycle_length, iteration):
        cycle_count = iteration // cycle_length
        if cycle_count % 2 == 0: # Up-cycle
            return min_lr + (max_lr-min_lr)*(iteration-cycle_count*cycle_length)/cycle_length
        else: # Down-cycle
            return max_lr - (max_lr-min_lr)*(iteration-cycle_count*cycle_length)/cycle_length
    
    def __get_increasing_cyclic_lr(self, min_lr, max_lr, starting_cycle_length, iteration):
        period_count = np.floor(np.log2(1+iteration/(2*starting_cycle_length)))
        current_period_length = 2*starting_cycle_length * 2**(period_count)
        period_start = 2*starting_cycle_length * (2**period_count - 1)

        if iteration < (period_start + current_period_length/2): # Up-cycle
            cycle_start = period_start
            print(f"Period: {period_count} started at {period_start}, Type: upcycle started at {cycle_start}")
            return min_lr + (max_lr-min_lr)*(iteration-cycle_start)/(current_period_length/2)
        else: # Down-cycle
            cycle_start = period_start + current_period_length/2
            print(f"Period: {period_count} started at {period_start}, Type: downcycle started at {cycle_start}")
            return max_lr - (max_lr-min_lr)*(iteration-cycle_start)/(current_period_length/2)

    def __softmax(self, y_pred: np.matrix):
        # y_pred will have dimension (10, 10000)
        return np.exp(y_pred) / np.sum(np.exp(y_pred), axis = 0)
    

    def __forward(self, Mx: np.ndarray = None):
        """Do a forward pass of the classifier with the loaded Mx (needs to be loaded!)"""
        x = Mx
        for layer in self.layers:
            x = layer.forward(x)
        return x


    def __cost_function(self, p: np.matrix, Y_batch: np.matrix):
        """Calculated the mean cross-entropy loss of a set of 
        predicted class probabilites"""
        # both have dimensions (10, batch_size)
        # We take the mean of the loss over the batch
        cross_entropy = self.__cross_entropy_loss(p, Y_batch)
        L2_Ws = self.lam*np.sum(self.W1 ** 2) + self.lam*np.sum(self.W2 ** 2)
        L2_filters = self.lam*np.sum(self.filters ** 2) + self.lam*np.sum(self.b0 ** 2)
        return cross_entropy + L2_Ws + L2_filters

    def __cross_entropy_loss(self, p: np.matrix, Y_batch: np.matrix):
        return np.mean(np.sum(-Y_batch*np.log(p), axis = 0))

    def __get_gradient(self, Mx_batch, Y_batch, P_batch):
        """Returns the gradient of the loss wrt weights and biases in a tuple. Will use self.s1 and self.h1"""
        
        g_batch = -np.transpose(Y_batch-P_batch) # (batch_size, 10)
        g_sum = np.expand_dims(g_batch.sum(axis = 0), axis = 1)
        batch_size = Mx_batch.shape[2] # nbr points in batch
        
        dl_dw2 = (g_batch.T @ self.h1.T)
        dl_db2 = g_sum
        
        g_batch = g_batch @ self.W2 # (batch_size, hid_dim)
        g_batch = g_batch * (self.s1 > 0).T
        g_sum = np.expand_dims(g_batch.sum(axis = 0), axis = 1)
        
        
        dl_dw1 = (g_batch.T @ self.h0.T)
        dl_db1 = g_sum

        
        g_batch = g_batch @ self.W1 
        g_batch = g_batch * (self.s0 > 0).T # (batch_size, n_p*nf)
        
        # Here it is important to first transpose in order to preserve sample indexing
        g_batch = g_batch.T.reshape((self.n_p, self.nf, batch_size), order='C')
        g_sum = np.sum(g_batch, axis=(0, 2)).reshape((self.nf, 1))
        Mxt = np.transpose(Mx_batch, (1, 0, 2))
        dl_dFsFlat = np.einsum('ijn, jln ->il', Mxt, g_batch, optimize=True)
        dl_dF_bias = g_sum

        dl_dFsFlat /= batch_size; dl_dF_bias /= batch_size
        dl_dw2 /= batch_size; dl_dw1 /= batch_size
        dl_db2 /= batch_size; dl_db1 /= batch_size

        dJ_dw2 = dl_dw2 + 2*self.lam*self.W2; dJ_dw1 = dl_dw1 + 2*self.lam*self.W1; 
        dJ_db2 = dl_db2; dJ_db1 = dl_db1

        dJ_dFs = dl_dFsFlat + 2*self.lam*self.filters
        dJ_dFBias = dl_dF_bias

        return {"W2": dJ_dw2, "W1": dJ_dw1, "b2": dJ_db2, "b1": dJ_db1, "Fs": dJ_dFs, "F_bias": dJ_dFBias}  


    def minibatchGD(self, X_train, Y_train, y_train, X_val = None, Y_val = None, y_val = None, num_epochs: int = 40, batch_size: int = 100, 
                    eta: float = .001, cyclic_lr: bool = True, cycles: int = 1, n_s: int = 500, shuffle_set: bool = True, show_plots: bool = True):
        n_datapoints = X_train.shape[1]
        num_minibatches = int(n_datapoints / batch_size)
        train_costs = []
        train_losses = []
        val_costs = []
        val_losses = []
        val_diffs = []
        lr_decay_epochs = []
        val_accuracys = []
        train_accuracys = []

        X_train_shuffled = X_train.copy()
        Y_train_shuffled = Y_train.copy()
        val_cost = np.inf
        
        lr = eta
        steps_since_plateau = 0
        pbar = tqdm.tqdm(range(num_epochs))
        cycle_index = 0

        for i in pbar:
            #print(f"Started epoch {i}")

            permutation = self.rng.permutation(n_datapoints)
            if shuffle_set:
                X_train_shuffled = X_train_shuffled[:, :, permutation]
                Y_train_shuffled = Y_train_shuffled[:, permutation]
    
            
            for j in range(num_minibatches):
                #print(f"minibatch {j}")
                j_start = j*batch_size
                j_end = (j+1)*batch_size

                X_batch = X_train_shuffled[:, :, j_start:j_end]
                P_batch = self.forward(X_batch)
                Y_batch = Y_train_shuffled[:, j_start:j_end]

                grads = self.get_gradient(X_batch, Y_batch, P_batch)

                if cyclic_lr:
                    lr = self.__get_cyclic_lr(1e-5, 1e-1, n_s, cycle_index)
                    #print(lr)
                    #print(f"lr: {lr}")
                    if cycle_index >= 2*cycles*n_s:
                        print(f"Breaking at cycle_index {cycle_index} and lr {lr}")
                        break # For Exercise 3 only
                
                self.filters -= lr*grads["Fs"]
                self.b0 -= lr*grads["F_bias"]
                self.W1 -= lr*grads["W1"]
                self.W2 -= lr*grads["W2"]
                self.b1 -= lr*grads["b1"]
                self.b2 -= lr*grads["b2"]
                cycle_index += 1
            
            P_train = self.forward(X_train)
            train_cost = self.cost_function(P_train, Y_train)
            train_loss = self.__cross_entropy_loss(P_train, Y_train)
            train_costs.append(train_cost)
            train_losses.append(train_loss)
            
            P_val = self.forward(X_val)
            val_cost = self.cost_function(P_val, Y_val)
            val_loss = self.__cross_entropy_loss(P_val, Y_val)
            val_costs.append(val_cost)
            val_losses.append(val_loss)
            
            if X_val is not None and y_val is not None and y_train is not None:
                val_accuracys.append(self.get_accuracy(X_val, y_val))
                train_accuracys.append(self.get_accuracy(X_train, y_train))

            
            if(len(val_costs)  >= 2):
                val_diffs.append(val_costs[-2]-val_costs[-1])
            
            
            if cyclic_lr and cycle_index >= 2*cycles*n_s:
                print(f"Breaking at cycle_index {cycle_index}")
                break # For Exercise 3 only
                
            pbar.set_postfix(loss = f"{round(val_loss, 4)}")

        if not show_plots: 
            return
        
        plt.plot(np.arange(i+1)*num_minibatches, train_costs, label = "Training Cost", color = "y")
        plt.plot(np.arange(i+1)*num_minibatches, train_losses, label = "Training Loss", linestyle='--', color = "r")
        plt.vlines(lr_decay_epochs, ymin = 1.5, ymax = 2, label = "Lr decay epochs")
        plt.plot(np.arange(i+1)*num_minibatches, val_costs, label = "Validation Cost", color = "b")
        plt.plot(np.arange(i+1)*num_minibatches, val_losses, label = "Validation Loss", linestyle = '--', color = "c")
        plt.xlabel("Update step")
        plt.legend()
        plt.title(f"Train/val cost + loss, lam={self.lam}")
        plt.show()

        plt.title("Train/Val accuracy over epochs")
        plt.plot(np.arange(i+1)*num_minibatches, val_accuracys, label="validation")
        plt.plot(np.arange(i+1)*num_minibatches, train_accuracys, label="training")
        plt.xlabel("Update step")
        plt.legend()
        plt.show()

    
    def get_accuracy(self, X, y):
        P = self.forward(X) # (10, batch_size)
        guesses = np.argmax(P, axis = 0)

        corrects = np.isclose(guesses, y).astype(np.int64)

        return np.sum(corrects) / X.shape[1]
    
    def display_accuracy_histogram(self, X, y):
        P = self.forward(X) # (10, batch_size)
        guesses = np.argmax(P, axis = 0)
        corrects = np.isclose(guesses, y)

        #print(corrects.shape)
        #print(corrects)

        class_corrects = np.array(y)[corrects]
        plt.hist(class_corrects)
        plt.show()

        return np.sum(corrects) / X.shape[1]
