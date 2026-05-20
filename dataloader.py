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
    

    def get_mirrored_data(self, X:np.ndarray) -> np.ndarray:
        """Mirrors the data horisontally (mirror line is vertical in the middle). 
        
        Args:
            X (np.ndarray): The data to be mirrored. (height, width, rgb, n)
            
        Returns:
            X_mirror (np.ndarray): The mirrored data. (height, width, rgb, n)"""
        
        return np.flip(X.copy(), 1)
    

    def get_rotated_data(self, X:np.ndarray) -> np.ndarray:
        """Get all the rotations of the original data. 
        
        Args:
            X (np.ndarray): The data to be mirrored. (height, width, rgb, n)
            
        Returns:
            X_rot (np.ndarray): The rotated data. (height, width, rgb, 4n)"""
        
        X_rot = X.copy()

        for i in range(1, 3+1):
            cur_X_rot = X.copy()
            X_rot = np.concat((X_rot, np.rot90(cur_X_rot, i)), axis=3)
        
        return X_rot


    def augment_data(self, X:np.ndarray) -> np.ndarray:
        """Augment the data by rotating and flipping. 
        
        Args:
            X (np.ndarray): The data to be augmented. (height, width, rgb, n)
            
        Returns:
            X_aug (np.ndarray): The augmented data. (height, width, rgb, 8n)"""
        
        X = X.copy()

        X = np.concat((X, self.get_mirrored_data(X)), axis=3)

        return self.get_rotated_data(X)


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
