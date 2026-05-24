import torch
import torchvision.transforms as trans
import numpy as np
import time
import pickle
from datasets import load_dataset, load_from_disk
import datasets
import matplotlib.pyplot as plt

class Dataloader:
    def __init__(self):
        self.X_mean:torch.Tensor = None
        """The mean of the training data. (height, width, rgb, 1)"""
        self.X_std:torch.Tensor = None
        """The standard deviation of the training data. (height, width, rgb, 1)"""
        self.augmentations = trans.Compose([
            trans.RandomResizedCrop(size=(64, 64), scale=(.5, 1), antialias=True),
            trans.RandomCrop(10),
            trans.RandomApply([
                trans.ColorJitter(brightness=(0, 0.5), contrast=(0, 0.5), saturation=(0, 0.5), hue=(0, 0.5))]),
            trans.RandomGrayscale(p=.2)
        ])

    def create_Mx(self, X: torch.Tensor, filter_width: int = 2) -> torch.Tensor:
        """
        Make the Mx representation of square images for use with convolutional filters.
        Args:
            X (torch.Tensor): The images to be turned into Mx. (height, width, rgb, n)
            filter_width (int): Width of square filter (in non-rgb direction)
            
        Returns:
            Mx (torch.Tensor): The Mx matrix. (width*height/f^2, rgb*f^2, n)"""
        
        height, width, rgb, n = X.shape
        f = filter_width

        if (height % f != 0) or (width % f != 0):
            raise ValueError("The height or width of the image isn't able to be divided by the filter. \
                             The height of the image is " + str(height) + ", the width is " + str(width) + " and f is " + str(f))

        X_blocks = X.reshape(height//f, f, width//f, f, rgb, n)

        X_permuted = X_blocks.permute(0, 2, 1, 3, 4, 5)

        Mx = X_permuted.reshape((height//f) * (width//f), rgb*f**2, n)
        
        self.Mx = Mx
        return Mx

    def mirror_data(self, X:torch.Tensor, mirror_prob:float):
        """Mirrors the data horisontally (mirror line is vertical in the middle). It is done in place. 
        
        Args:
            X (torch.Tensor): The data to be mirrored. (height, width, rgb, n)
            mirror_prob (float): Probability of flipping/mirroring each image. [0, 1]"""
        
        if mirror_prob == 0:
            return
        
        n = X.shape[-1]

        mirr_ind = torch.where(torch.rand(n) < mirror_prob)[0] #All images to be mirrored

        if len(mirr_ind) == 0:
            return
        
        X[..., mirr_ind] = torch.flip(X[..., mirr_ind], [1])
    

    def rotate_data(self, X:torch.Tensor, rot_prob:float):
        """Rotate the data in place. 
        
        Args:
            X (torch.Tensor): The data to be mirrored. (height, width, rgb, n)
            rot_prob (float): Probability of rotating each image. [0, 1]"""

        if rot_prob == 0:
            return
        
        n = X.shape[-1]
        n_rot = 3

        rot_probs = torch.rand(n)
        rot_deg = torch.randint(1, n_rot+1, size=(n,))

        for i in range(1, n_rot+1):
            rot_ind = torch.where((rot_probs < rot_prob) & (rot_deg == i))[0] #All images to be rotated that way
            
            if len(rot_ind) > 0:
                X[..., rot_ind] = torch.rot90(X[..., rot_ind], i)


    def add_noise(self, X:torch.Tensor, noise_prob:float, noise_std:float):
        """Adds noise in place. 
        
        Args:
            X (torch.Tensor): The data to be mirrored. (height, width, rgb, n)
            noise_prob (float): Probability of adding noise to each image. [0, 1]
            noise_std (float): Standard deviation of the noise. """
        
        if noise_prob == 0:
            return
        
        n = X.shape[-1]

        noise_ind = torch.where(torch.rand(n) < noise_prob)[0]

        if len(noise_ind) == 0:
            return
        
        noise_batch = X[..., noise_ind]

        max_value = 255.0
        min_value = 0.0
        noise_std *= max_value
        if (self.X_mean is not None) and (self.X_std is not None):
            max_value = (max_value - self.X_mean) / self.X_std
            min_value = (min_value - self.X_mean) / self.X_std
            noise_std = noise_std / self.X_std
       

        noise = torch.randn_like(noise_batch) * noise_std

        X[..., noise_ind] = torch.clamp(noise_batch + noise, min=min_value, max=max_value)


    def augment_data(self, X:torch.Tensor, flip_prob:float=0.5, rot_prob:float=0.1, noise_prob:float=0.2, \
                     noise_std:float=0.05) -> torch.Tensor:
        """Augment the data by rotating, flipping and adding noise. Clones the data. 
        
        Args:
            X (torch.Tensor): The data to be augmented. (height, width, rgb, n)
            flip_prob (float): Probability of flipping/mirroring each image. [0, 1]
            rot_prob (float):  Probability of rotating each image -90, 90 or 180 degrees. [0, 1]
            noise_prob (float): Probability of adding noise to each image. [0, 1]
            noise_std (float): Standard deviation of the noise. 
            
        Returns:
            X_aug (torch.Tensor): The augmented data. (height, width, rgb, n)"""
        
        X_aug = X.clone()
        # Own augmentation implementations
        self.add_noise(X=X_aug, noise_prob=noise_prob, noise_std=noise_std)
        self.mirror_data(X=X_aug, mirror_prob=flip_prob)
        self.rotate_data(X=X_aug, rot_prob=rot_prob)

        # Torchvision augmentation
        X_aug_perm = X_aug.permute(3, 2, 0, 1)
        self.augmentations(X_aug_perm)
        X_aug = X_aug_perm.permute(2, 3, 1, 0)

        return X_aug


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
    

    def load_train_data(self, train_lim:int | None=None,print_progress:bool=False) -> dict:
        """Load the train and validation data from tiny imagenet. 
        
        Args:
            train_lim (int | None): None if all training data should be loaded, \
                otherwise specifiy how much. 
            print_progress (bool): Set to true if you want prints showing the progress

        Returns:
            train_val_data (dict): Dictionary of the training and validation data and labels. 
                - **'X_train'** (torch.Tensor): The training data. (height, width, rgb, n_train)
                - **'Y_train'** (torch.Tensor): The one-hot training label matrix. (n_labels, n_train)
                - **'y_train'** (torch.Tensor): The training labels. (n_train,)
                - **'X_val'** (torch.Tensor): The validation data. (height, width, rgb, n_val)
                - **'Y_val'** (torch.Tensor): The one-hot validation label matrix. (n_labels, n_val)
                - **'y_val'** (torch.Tensor): The validation labels. (n_val,)"""
        if print_progress: print("Before loading")
        ds = load_from_disk("Datasets/split_tiny_imagenet")
        if print_progress: print("Loaded from disk")
        n_classes = 200

        train_val_data = {}

        train_images = ds['train']['image']
        height, width, rgb = np.array(train_images[0].convert('RGB')).shape

        # -- Train --
        if print_progress: print("Started with train")
        n_train = len(train_images)
        if train_lim is None:
            X_train = torch.empty((height, width, rgb, n_train), dtype=torch.float32)
            for i, img in enumerate(train_images):
                X_train[..., i] = torch.from_numpy(np.array(img.convert('RGB')))
            train_val_data['y_train'] = torch.tensor(ds['train']['label'], dtype=torch.long)
    
        else:
            X_train = torch.empty((height, width, rgb, train_lim), dtype=torch.float32)
            for i, img in enumerate(train_images):
                if i >= train_lim:
                    break
                X_train[..., i] = torch.from_numpy(np.array(img.convert('RGB')))

            train_val_data['y_train'] = torch.tensor(ds['train']['label'][:train_lim], dtype=torch.long)

        train_val_data['X_train'] = X_train
        
        train_val_data['Y_train'] = torch.nn.functional.one_hot(train_val_data['y_train'], num_classes=n_classes).float().T
        train_val_data['y_train'] = train_val_data['y_train'].float()

        # -- Val --
        if print_progress: print("Started with val")
        val_images = ds['val']['image']
        n_val = len(val_images)
        X_val = torch.empty((height, width, rgb, n_val), dtype=torch.float32)
        for i, img in enumerate(val_images):
            X_val[..., i] = torch.from_numpy(np.array(img.convert('RGB')))
        train_val_data['X_val'] = X_val
        
        train_val_data['y_val'] = torch.tensor(ds['val']['label'], dtype=torch.long)
        
        train_val_data['Y_val'] = torch.nn.functional.one_hot(train_val_data['y_val'], num_classes=n_classes).float().T
        train_val_data['y_val'] = train_val_data['y_val'].float()
        if print_progress: print("Finished loading all the data!")
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

    def show_filters(self, F: torch.Tensor, freeze: bool = False):
        F_visual = F.detach().clone()
        nf = F.shape[1]
        f = int(torch.math.sqrt(F.shape[0]/3))
        min_val = torch.min(F_visual).item()
        max_val = torch.max(F_visual).item()
        F_visual = F_visual.reshape((f, f, 3, nf))

        rows = 8
        cols = 8
        fig, axs = plt.subplots(cols, rows, figsize=(10, 5))
        for i in range(rows*cols):
            im = F_visual[..., i] - min_val
            im /= (max_val-min_val + 1e-8)
            axs[i // rows, i%rows].imshow(im)
            axs[i // rows, i%rows].axis('off')
        
        if freeze: plt.show()
        else:
            plt.pause(0.1)
    
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
        
        if (self.X_mean is not None) and (self.X_std is not None):
            X_im = X * self.X_std + self.X_mean #Undo normalization
        else:
            X_im = X

        X_im /= 255.0
        fig, axs = plt.subplots(1, max(n, 2), figsize=(10, 5))
        for i in range(n):
            axs[i].imshow(X_im[:, :, :, i])
            axs[i].axis('off')
        plt.show()

    def normalize(self, X: torch.Tensor, use_train_data:bool=False) -> torch.Tensor:
        """Normalize the data. Also saves the mean and standard deviation in the class. 
        
        Args:
            X (torch.Tensor): The images. (height, width, rgb, n)
            use_train_data (bool): Should be true if the data to be normalized is not the training data. \
                Make sure that you've already normalized the train data in this class to utilize its mean and standard deviation. 
        
        Returns:
            X_norm (torch.Tensor): The normalized data. (height, width, rgb, n)"""
        
        if not use_train_data:
            
            self.X_mean = torch.mean(X, dim = 3).unsqueeze(dim=3) # (height, width, rgb, 1)
            self.X_std = torch.std(X, dim = 3).unsqueeze(dim=3) # (height, width, rgb, 1)
        
        return (X - self.X_mean) / self.X_std
