import numpy as np
import pickle
import matplotlib.pyplot as plt

class Dataloader:
    def __init__(self):
        pass
    
    def create_Mx(self, X: np.ndarray, filter_width: int = 2, image_width: int = 32):
        """
        Make the Mx representation of square images for use with convolutional filters.
        Args:
            filter_width: Width of square filter (in non-rgb direction)
            image_width: Width of square image in the space (in non-rgb direction)"""
        f = filter_width
        X_ims = np.transpose(X.reshape((image_width, image_width, 3, X.shape[1]), order='F'), (1, 0, 2, 3))
        ni = X_ims.shape[3]
        Mx = np.zeros(shape=((image_width//f)**2, f*f*3, ni))
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
    
    def unpickle(self, file: str, folder: str):
        try:
            with open(folder + file, "rb") as fi:
                data = pickle.load(fi, encoding="bytes")
        except:
            print("file does not seem to exist?")
            return None
        return data

    def load_dataset(self, file: str, folder: str = ""):
            """Tries to returns (X, Y, y) from dataset
            Args:
                file: the filename of the dataset-batch
                folder: the folder that the file lies in from command lines perspective"""
            dataset: np.matrix = self.unpickle(file, folder)
            if dataset == None: 
                raise ValueError("Could not load dataset, returning null...")
                return None
            
            X = dataset[b'data'].astype(np.float64) / 255
            X = X.transpose()

            y = dataset[b"labels"]
            Y = np.eye(10)[y]
            Y = Y.transpose()

            return (X, Y, y)

    
    def show_images(self, X, n=5):
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
