import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import tqdm

class Convolution_Layer:
    def __init__(self):
        pass

K = 10 # nbr classes
class CNN:
    def __init__(self, hid_dim: int, lam: float = 0, num_filters: int = 2, f:int = 4):
        """Creates a linear classifier with 10 output classes."""
        
        rng = np.random.default_rng()
        seed = 42
        bit_gen = type(rng.bit_generator)
        rng.bit_generator.state = bit_gen(seed).state
        self.rng = rng
        self.n_p = (32//f)**2
        
        # He initialization
        self.filters = rng.normal(loc=0, scale=np.sqrt(2/(f*f*3)), size=(f*f*3, num_filters))
        self.nf = num_filters
        self.b0 = np.zeros(shape = (num_filters, 1))
        
        
        self.W1 = np.sqrt(2/(self.n_p*self.nf))*rng.standard_normal(size = (hid_dim, self.n_p*self.nf)) # (hid, n_p*nf)
        self.b1 = np.zeros(shape = (hid_dim, 1)) # (hid, 1)
        self.W2 = np.sqrt(2/hid_dim)*rng.standard_normal(size = (K, hid_dim)) # (10, hid)
        self.b2 = np.zeros(shape = (K, 1)) # (10, hid)

        self.s0 = None
        self.h0 = None
        self.s1 = None
        self.h1 = None
        self.s2 = None
        self.lam = lam

    def load_Mx(self, X: np.ndarray):
        X_ims = np.transpose(X.reshape((32, 32, 3, X.shape[1]), order='F'), (1, 0, 2, 3))
        ni = X_ims.shape[3]
        Mx = np.zeros(shape=(n_p, f*f*3, ni))
        for img_index in range(ni):
            img = X_ims[:, :, :, img_index]
            l = 0
            for i in range(32//f):
                for j in range(32//f):
                    patch = img[i*f:i*f+f, j*f:j*f+f, :]
                    Mx[l, :, img_index] = patch.reshape((1, f*f*3), order='C')
                    l += 1
        
        self.Mx = Mx
        return Mx
    
    def __get_cyclic_lr(self, min_lr, max_lr, cycle_length, iteration):
        cycle_count = iteration // cycle_length
        if cycle_count % 2 == 0: # Up-cycle
            return min_lr + (max_lr-min_lr)*(iteration-cycle_count*cycle_length)/cycle_length
        else: # Down-cycle
            return max_lr - (max_lr-min_lr)*(iteration-cycle_count*cycle_length)/cycle_length
    
    def get_increasing_cyclic_lr(self, min_lr, max_lr, starting_cycle_length, iteration):
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

    def ReLU(self, x):
        x[x < 0] *= 0
        return x
    

    def forward(self, Mx: np.ndarray = None):
        """Do a forward pass of the classifier with the loaded Mx (needs to be loaded!)"""
        # Convolutional


        # Mx has dimension (n_p, f*f*3, n)
        # Fs_flat has dimensions (f*f*3, nf)
    
        conv_res = np.einsum("ijn, jl -> iln", Mx, self.filters, optimize=True) #(np, nf, batch_size)
        conv_res += self.b0.reshape(1, self.nf, 1)
        #print(f"First row of s0 analytical: {conv_res[:, 0, 0]}")
        self.s0 = conv_res.reshape((self.n_p*self.nf, conv_res.shape[2]), order='C') # (n_p*n_f, batch_size)
        self.h0 = self.ReLU(self.s0)

        # Fully connected
        self.s1 = self.W1 @ self.h0 + self.b1 # (hid, batch_size)
        self.h1 = self.ReLU(self.s1)
        self.s2 = self.W2 @ self.h1 + self.b2 # (out, batch_size)
        return self.__softmax(self.s2)

    def cost_function(self, p: np.matrix, Y_batch: np.matrix):
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

    def get_gradient(self, Mx_batch, Y_batch, P_batch):
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

        print(corrects.shape)
        print(corrects)

        class_corrects = np.array(y)[corrects]
        plt.hist(class_corrects)
        plt.show()


        return np.sum(corrects) / X.shape[1]



class Data_Loader:
    def __init__(self):
        pass

    def unpickle(file: str):
        try:
            with open("Datasets/cifar-10-batches-py/" + file, "rb") as fi:
                data = pickle.load(fi, encoding="bytes")
        except:
            print("file does not seem to exist?")
            return None
        return data

    def load_dataset(self, file: str):
            """Returns (X, Y, y)"""
            dataset: np.matrix = self.unpickle(file)
            if dataset == None: return None
            
            X = dataset[b'data'].astype(np.float64) / 255
            X = X.transpose()

            y = dataset[b"labels"]
            Y = np.eye(10)[y]
            Y = Y.transpose()

            return (X, Y, y)

    
    def show_images(X, n=5):
        X_im = X.reshape((32, 32, 3, X.shape[1]), order='F')
        X_im = np.transpose(X_im, (1, 0, 2, 3))

        
        fig, axs = plt.subplots(1, 5, figsize=(10, 5))
        for i in range(n):
            axs[i].imshow(X_im[:, :, :, i])
            axs[i].axis('off')
        plt.pause(0.1)

    def normalize(X: np.matrix):
        X_mean = np.resize(np.mean(X, axis = 1), (X.shape[0], 1))
        X_std = np.resize(np.std(X, axis = 1), (X.shape[0], 1))
        return (X - X_mean) / X_std
