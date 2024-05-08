import pickle
from typing import Tuple
import pickle
import numpy as np


from dlvc.datasets.dataset import  Subset, ClassificationDataset

import pickle
from typing import Tuple
import pickle
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import os

from dlvc.datasets.dataset import  Subset, ClassificationDataset

class CIFAR10Dataset(ClassificationDataset):
    '''
    Custom CIFAR-10 Dataset.
    '''

    def __init__(self, fdir: str, subset: Subset, transform=None):
        '''
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.
        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        '''

        if not os.path.isdir(fdir):
            raise ValueError(f"{fdir} is not a directory")

        self.fdir = fdir
        self.subset = subset
        self.transform = transform
        self.classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self._int_to_class = { i:label for i, label in enumerate(self.classes)}
        
        self.training_files = [f"data_batch_{i}" for i in range(1,5)]
        self.validation_files = ["data_batch_5"]
        self.test_files = ["test_batch"]
        self.data = np.ones((40000,32,32,3))
        self.labels = []
        
        files_to_process = self.training_files
        if subset.name == Subset.VALIDATION.name:
            files_to_process = self.validation_files
            self.data = np.ones((10000,32,32,3))
        elif subset.name == Subset.TEST.name:
            files_to_process = self.test_files
            self.data = np.ones((10000,32,32,3))

        for i, file in enumerate(files_to_process):
            file_path = f"{self.fdir}/{file}"
            if not os.path.isfile(file_path):
                raise ValueError(f"{file} does not exist")
            images = self._unpickle(file_path)
            data = images[b"data"].reshape(-1, 3, 32, 32)
            data = np.moveaxis(data, 1, -1)
            self.data[i*10000:(i+1)*10000] = data
            self.labels += images[b"labels"]
            break

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        '''
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        '''
        if idx < 0 or idx >= self.__len__():
            raise IndexError
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return (img, self.labels[idx])
    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        counts = Counter(self.labels)
        return { self._int_to_class[key]:val for key, val in counts.items()}

    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

