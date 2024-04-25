import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# for wandb users:
#from dlvc.wandb_logger import WandBLogger

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Holds training logic.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds validation logic for one epoch.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds training logic for one epoch.
        '''

        pass

class ImgClassificationTrainer(BaseTrainer):
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
                 val_frequency: int = 5) -> None:
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
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
        self.best_per_class_accuracy = 0.0

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        self.train_metric.reset()
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            self.train_metric.update(output, target)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
        self.lr_scheduler.step()  # adjust learning rate every 5 batches
        loss, accuracy, class_accuracy = loss.item(), self.train_metric.accuracy(), self.train_metric.per_class_accuracy()
        print(f"\nTRAIN, EPOCH: {epoch_idx} \nLoss: {loss}\nAccuracy: {accuracy}\nClass Accuracy: {class_accuracy}")
        return loss, accuracy, class_accuracy

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        self.model.eval()
        self.val_metric.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.validation_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                self.val_metric.update(output, target)
                loss = self.loss_fn(output, target).item()
        accuracy = self.val_metric.accuracy()
        class_accuracy = self.val_metric.per_class_accuracy()
        print(
            f"\nVALIDATION, EPOCH: {epoch_idx} \nLoss: {loss}\nAccuracy: {accuracy}\nClass Accuracy: {class_accuracy}")
        return loss, accuracy, class_accuracy

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        for i in range(0, self.num_epochs):
            train_metrics = self._train_epoch(i)
            self.metrics_train.append(train_metrics)
            if i % self.val_frequency == 0:
                eval_metrics = self._val_epoch(i)
                self.metrics_val.append(eval_metrics)
                if eval_metrics[2] > self.best_per_class_accuracy:
                    print("Best mean per class accuracy on validation data set is higher. Saving new best model")
                    torch.save(self.model.state_dict(), self.training_save_dir)