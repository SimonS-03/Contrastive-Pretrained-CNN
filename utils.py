import torch
import math
import matplotlib.pyplot as plt

class lr_schedule:
    def __init__(self):
        pass
    
    def get_lr(self, iteration):
        raise NotImplementedError("get_lr method not implemented in the lr-schedule!")
    
    def plot_lr(self, max_iterations):
        its = torch.arange(max_iterations)
        lrs = []
        for i in its:
            lrs.append(self.get_lr(i))
        
        plt.plot(its, lrs)
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.show()

class constant_lr(lr_schedule):
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr

    def get_lr(self, iteration):
        return self.lr



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

    def get_lr(self, iteration):
        period_count = math.floor(math.log2(1+iteration/(2*self.starting_cycle_length)))
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
    
    def update_gradient(self, params: list[dict[str, float]], grads: list[dict[str, float]], iteration: int, lam: float) -> list[dict[str, float]]:
        """Returns the gradients according to the AdamW optimizer. 
        
        Args:
            params (list[dict[str, float]]): List of the parameters of W and b of each layer. 
            grads (list[dict[str, float]]): List of the calculated gradients of W and b of each layer without using this optimizer. 
            iteration (int): The iteration the optimizer is in. 
            lam (float): Not used in this optimizer. 
        
        Returns:
            new_grads (list[dict[str, float]]): List of the new gradients of W and b of each layer using this optimizer. """

        new_grads = [{"W": 0, "b": 0} for grad in grads] 
        if self.mt is None or self.vt is None:
            self.mt = [{"W": 0, "b": 0} for grad in grads] 
            self.vt = [{"W": 0, "b": 0} for grad in grads] 
        
        for layer in range(len(grads)):
            layer_grads: dict = grads[layer]
            for param in layer_grads.keys():
                param_grad = layer_grads[param].clone()
                self.mt[layer][param] = self.b1*self.mt[layer][param] + (1-self.b1)*param_grad
                self.vt[layer][param] = self.b2*self.vt[layer][param] + (1-self.b2)*param_grad**2
                mt_hat = self.mt[layer][param] / (1-self.b1**iteration + self.eps)
                vt_hat = self.vt[layer][param] / (1-self.b2**iteration + self.eps)
                new_grads[layer][param] = mt_hat / (torch.sqrt(vt_hat) + self.eps) + 2*lam*params[layer][param]
                #print(f"Grad-norm in layer {layer} and param {param}: {new_grads[layer][param].norm()}")

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

def softmax(y_pred: torch.Tensor):
        # y_pred will have dimension (classes, n_imgs)
        return torch.exp(y_pred) / torch.sum(torch.exp(y_pred), axis = 0)


def dropout(activations: torch.Tensor, prob: float):
    if prob == 0:
        return activations.clone()
    elif prob == 1:
        return torch.zeros_like(activations)
    

    if prob > 1 or prob < 0:
        raise ValueError(f"Dropout likelihood of {prob} is not in [0, 1]!")

    mask = torch.rand_like(activations) >= prob
    return activations*mask / (1-prob)
    
