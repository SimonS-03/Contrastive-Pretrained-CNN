import numpy as np
import matplotlib.pyplot as plt

class lr_schedule:
    def __init__(self):
        pass
    
    def get_lr(self, iteration):
        raise NotImplementedError("get_lr method not implemented in the lr-schedule!")
    
    def plot_lr(self, max_iterations):
        its = np.arange(max_iterations)
        lrs = []
        for i in its:
            lrs.append(self.get_lr(i))
        
        plt.plot(its, lrs)
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.show()


class cyclic_lr(lr_schedule):
    def __init__(self, cycle_length: float, min_lr: float, max_lr: float):
        super().__init__()
        self.cycle_length = cycle_length
        self.min_lr = min_lr
        self.max_lr = max_lr

    def get_lr(self, iteration):
        cycle_count = iteration // self.cycle_length
        if cycle_count % 2 == 0: # Up-cycle
            return self.min_lr + (self.max_lr-self.min_lr)*(iteration-cycle_count*self.cycle_length)/self.cycle_length
        else: # Down-cycle
            return self.max_lr - (self.max_lr-self.min_lr)*(iteration-cycle_count*self.cycle_length)/self.cycle_length


class cyclic_increasing_lr(lr_schedule):
    def __init__(self, starting_cycle_length: float, min_lr: float, max_lr: float):
        self.starting_cycle_length = starting_cycle_length
        self.min_lr = min_lr
        self.max_lr = max_lr

    def get_increasing_cyclic_lr(self, iteration):
        period_count = np.floor(np.log2(1+iteration/(2*self.starting_cycle_length)))
        current_period_length = 2*self.starting_cycle_length * 2**(period_count)
        period_start = 2*self.starting_cycle_length * (2**period_count - 1)

        if iteration < (period_start + current_period_length/2): # Up-cycle
            cycle_start = period_start
            return self.min_lr + (self.max_lr-self.min_lr)*(iteration-cycle_start)/(current_period_length/2)
        else: # Down-cycle
            cycle_start = period_start + current_period_length/2
            return self.max_lr - (self.max_lr-self.min_lr)*(iteration-cycle_start)/(current_period_length/2)
    

class optimizer:
    def __init__(self):
        pass

    def update_gradient(self, params: list[dict[str, float]], grads: list[dict[str, float]], iteration: int, lam: float):
        raise NotImplementedError("get_grad method not implemented in optimizer")

class AdamW(optimizer):
    """The AdamW optimizer implements momentum along with a running 
    square average of past gradients. It also decouples L2 reg from adaptive lr. """
    def __init__(self, b1 = .9, b2 = .999, eps: float = 1e-8):
        super().__init__()
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.mt: list[dict[str, float]] = None
        self.vt: list[dict[str, float]] = None
    
    def update_gradient(self, params: list[dict[str, float]], grads: list[dict[str, float]], iteration: int, lam: float):
        new_grads = [{"W": 0, "b": 0} for grad in grads] 
        if self.mt is None or self.vt is None:
            self.mt = [{"W": 0, "b": 0} for grad in grads] 
            self.vt = [{"W": 0, "b": 0} for grad in grads] 
        
        for layer in range(len(grads)):
            layer_grads: dict = grads[layer]
            for param in layer_grads.keys():
                self.mt[layer][param] = self.b1*self.mt[layer][param] + (1-self.b1)*layer_grads[param]
                self.vt[layer][param] = self.b2*self.vt[layer][param] + (1-self.b2)*layer_grads[param]**2
                mt_hat = self.mt[layer][param] / (1-self.b1**iteration)
                vt_hat = self.vt[layer][param] / (1-self.b2**iteration)
                new_grads[layer][param] = (mt_hat / (np.sqrt(vt_hat) + self.eps) + 2*lam*params[layer][param])

        return new_grads

class Identity(optimizer):
    """The Identity optimizer implements only L2 regularization."""
    def __init__(self):
        super().__init__()
    
    def update_gradient(self, params: list[dict[str, float]], grads: list[dict[str, float]], iteration: int, lam: float):
        new_grads = [{"W": 0, "b": 0} for grad in grads] 
        for layer in range(len(grads)):
            layer_grads: dict = grads[layer]
            for param in layer_grads.keys():
                new_grads[layer][param] = grads[layer][param] + 2*lam*params[layer][param]
        
        return new_grads

def softmax(y_pred: np.matrix):
        # y_pred will have dimension (classes, n_imgs)
        return np.exp(y_pred) / np.sum(np.exp(y_pred), axis = 0)
    
