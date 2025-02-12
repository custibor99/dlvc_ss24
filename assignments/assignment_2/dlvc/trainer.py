import collections
import torch
from typing import  Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader


#from dlvc.wandb_logger import WandBLogger

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

class ImgSemSegTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5):
        
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
            val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
            train_data (dlvc.datasets...): Train dataset
            val_data (dlvc.datasets...): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch 
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        '''
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.train_data, self.val_data = train_data, val_data

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        self.metrics_val = []
        self.metrics_train = []
        self.best_metric = 0.0
        self.train_epoch_steps = np.ceil(len(train_data) / batch_size)
        self.val_epoch_steps = np.ceil(len(train_data) / batch_size)

    
        ##TODO implement
        # recycle your code from assignment 1 or use/adapt reference implementation
        pass
        

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.

        epoch_idx (int): Current epoch number
        """
        ##TODO implement
        # recycle your code from assignment 1 or use/adapt reference implementation
        self.train_metric.reset()
        self.model.train()
        for batch_idx, (data, target) in tqdm(enumerate(self.train_loader),desc="train epoch", total=self.train_epoch_steps):
            b, c, h, w = target.shape
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output,target.view(b,h,w))
            self.train_metric.update(output,target.view(b,h,w))
            loss.backward()
            self.optimizer.step()
        self.lr_scheduler.step()  # adjust learning rate every 5 batches
        loss, accuracy = loss.item(), self.train_metric.mIoU(),
        print(f"\nTRAIN, EPOCH: {epoch_idx} \nLoss: {loss}\mIoU: {accuracy}")
        return loss, accuracy


    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        ##TODO implement
        # recycle your code from assignment 1 or use/adapt reference implementation
        self.model.eval()
        self.val_metric.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.validation_loader), desc="val epoch", total=self.val_epoch_steps, ):
                b, c, h, w = target.shape
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                self.val_metric.update(output, target.view(b,h,w))
                loss = self.loss_fn(output,target.view(b,h,w)).item()
        accuracy = self.val_metric.mIoU()
        print(
            f"\nVALIDATION, EPOCH: {epoch_idx} \nLoss: {loss}\mIoU: {accuracy}")
        return loss, accuracy

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean IoU on validation data set is higher
        than currently saved best mean IoU or if it is end of training. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        ##TODO implement
        # recycle your code from assignment 1 or use/adapt reference implementation
        for i in range(0, self.num_epochs):
            train_metrics = self._train_epoch(i)
            self.metrics_train.append(train_metrics)
            if i % self.val_frequency == 0:
                eval_metrics = self._val_epoch(i)
                self.metrics_val.append(eval_metrics)
                if eval_metrics[1] > self.best_metric:
                    print("New metric is higher. Saving new best model")
                    #torch.save(self.model.state_dict(), self.training_save_dir)
                    self.best_metric = eval_metrics[1]

                





            
            


