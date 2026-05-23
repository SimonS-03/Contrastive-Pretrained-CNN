import torch
import numpy as np
import time
import pickle
from datasets import load_dataset, load_from_disk
import datasets
import matplotlib.pyplot as plt

class Dataloader:
    def __init__(self):
        pass
    
    def create_Mx(self, X: torch.Tensor, filter_width: int = 2, image_width: int = 32):
        """
        Make the Mx representation of square images for use with convolutional filters.
        Args:
            filter_width: Width of square filter (in non-rgb direction)
            image_width: Width of square image in the space (in non-rgb direction)"""
        f = filter_width
        X_ims = X.T.reshape(X.shape[1], 3, image_width, image_width).permute(2, 3, 1, 0)
        ni = X_ims.shape[3]
        Mx = torch.zeros(size=((image_width//f)**2, f*f*3, ni))
        for img_index in range(ni):
            img = X_ims[:, :, :, img_index]
            l = 0
            for i in range(32//f):
                for j in range(32//f):
                    patch = img[i*f:i*f+f, j*f:j*f+f, :]
                    Mx[l, :, img_index] = patch.reshape(1, f*f*3)
                    l += 1
        
        self.Mx = Mx
        return Mx
    

    def get_mirrored_data(self, X:torch.Tensor) -> torch.Tensor:
        """Mirrors the data horisontally (mirror line is vertical in the middle). 
        
        Args:
            X (torch.Tensor): The data to be mirrored. (height, width, rgb, n)
            
        Returns:
            X_mirror (torch.Tensor): The mirrored data. (height, width, rgb, n)"""
        
        return torch.flip(X.clone(), [1])
    

    def get_rotated_data(self, X:torch.Tensor) -> torch.Tensor:
        """Get all the rotations of the original data. 
        
        Args:
            X (torch.Tensor): The data to be mirrored. (height, width, rgb, n)
            
        Returns:
            X_rot (torch.Tensor): The rotated data. (height, width, rgb, 4n)"""
        
        X_rot = X.clone()

        for i in range(1, 3+1):
            cur_X_rot = X.clone()
            X_rot = torch.concat((X_rot, torch.rot90(cur_X_rot, i)), axis=3)
        
        return X_rot


    def augment_data(self, X:torch.Tensor) -> torch.Tensor:
        """Augment the data by rotating and flipping. 
        
        Args:
            X (torch.Tensor): The data to be augmented. (height, width, rgb, n)
            
        Returns:
            X_aug (torch.Tensor): The augmented data. (height, width, rgb, 8n)"""
        
        X = X.clone()

        X = torch.concat((X, self.get_mirrored_data(X)), axis=3)

        return self.get_rotated_data(X)


    def unpickle(self, file: str, folder: str):
        try:
            with open(folder + file, "rb") as fi:
                data = pickle.load(fi, encoding="bytes")
        except:
            print("file does not seem to exist?")
            return None
        return data


    def import_tiny_imagenet(self) -> None:
        """USE ONLY ONCE! Import and save the tiny imagenet dataset."""

        ds = load_dataset("zh-plus/tiny-imagenet")

        ds.save_to_disk("Datasets/tiny_imagenet")

    
    def split_tiny_imagenet(self):
        """USE ONLY ONCE! Split the tiny imagenet dataset into train, validation and test data. \
            Ensures an equal proportion of labels in each set. """

        ds = load_from_disk("Datasets/tiny_imagenet")

        train_val_split = ds["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=42)

        final_ds = datasets.DatasetDict({
            'train': train_val_split['train'],
            'val': train_val_split['test'],
            'test': ds['valid']
        })

        print(final_ds)

        final_ds.save_to_disk("Datasets/split_tiny_imagenet")
    

    def load_train_data(self) -> dict:
        """Load the train and validation data from tiny imagenet. 
        
        Returns:
            train_val_data (dict): Dictionary of the training and validation data and labels. 
                - **'X_train'** (torch.Tensor): The training data. (height, width, rgb, n_train)
                - **'Y_train'** (torch.Tensor): The one-hot training label matrix. (n_labels, n_train)
                - **'y_train'** (torch.Tensor): The training labels. (n_train,)
                - **'X_val'** (torch.Tensor): The validation data. (height, width, rgb, n_val)
                - **'Y_val'** (torch.Tensor): The one-hot validation label matrix. (n_labels, n_val)
                - **'y_val'** (torch.Tensor): The validation labels. (n_val,)"""
        ds = load_from_disk("Datasets/split_tiny_imagenet")
        
        n_classes = 200

        train_val_data = {}

        train_images = ds['train']['image']
        height, width, rgb = np.array(train_images[0].convert('RGB')).shape

        # -- Train --

        n_train = len(train_images)
        X_train = torch.empty((height, width, rgb, n_train), dtype=torch.float32)
        for i, img in enumerate(train_images):
            X_train[..., i] = torch.from_numpy(np.array(img.convert('RGB')))
        train_val_data['X_train'] = X_train
        
        train_val_data['y_train'] = torch.tensor(ds['train']['label'], dtype=torch.long)
        
        train_val_data['Y_train'] = torch.nn.functional.one_hot(train_val_data['y_train'], num_classes=n_classes)
       
        # -- Val --

        val_images = ds['val']['image']
        n_val = len(val_images)
        X_val = torch.empty((height, width, rgb, n_val), dtype=torch.float32)
        for i, img in enumerate(val_images):
            X_val[..., i] = torch.from_numpy(np.array(img.convert('RGB')))
        train_val_data['X_val'] = X_val
        
        train_val_data['y_val'] = torch.tensor(ds['val']['label'], dtype=torch.long)
        
        train_val_data['Y_val'] = torch.nn.functional.one_hot(train_val_data['y_val'], num_classes=n_classes)
        
        return train_val_data


    def load_dataset(self, file: str, folder: str = ""):
            """Tries to returns (X, Y, y) from dataset
            Args:
                file: the filename of the dataset-batch
                folder: the folder that the file lies in from command lines perspective"""
            dataset: torch.Tensor = self.unpickle(file, folder)
            if dataset == None: 
                raise ValueError("Could not load dataset, returning null...")
                return None
            
            X = dataset[b'data'].astype(torch.float64) / 255
            X = X.transpose()

            y = dataset[b"labels"]
            Y = torch.eye(10)[y]
            Y = Y.transpose()

            return (X, Y, y)

    
    def show_images(self, X, n=5):
        X_im = X.reshape((32, 32, 3, X.shape[1]), order='F')
        X_im = torch.transpose(X_im, (1, 0, 2, 3))

        
        fig, axs = plt.subplots(1, 5, figsize=(10, 5))
        for i in range(n):
            axs[i].imshow(X_im[:, :, :, i])
            axs[i].axis('off')
        plt.pause(0.1)

    def show_tiny_images(self, X, n=5):
        """Visualize some of the images from Tiny ImageNet. 
        
        Args:
            X (torch.Tensor): The images. (height, width, rgb, n)"""
        
        X_im = X / 255.0
        fig, axs = plt.subplots(1, 5, figsize=(10, 5))
        for i in range(n):
            axs[i].imshow(X_im[:, :, :, i])
            axs[i].axis('off')
        plt.show()

    def normalize(X: torch.Tensor):
        X_mean = torch.resize(torch.mean(X, axis = 1), (X.shape[0], 1))
        X_std = torch.resize(torch.std(X, axis = 1), (X.shape[0], 1))
        return (X - X_mean) / X_std, X_mean, X_std
