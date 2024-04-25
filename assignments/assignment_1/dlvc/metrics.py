from abc import ABCMeta, abstractmethod
import torch
from statistics import mean

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass



class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self, classes: tuple[str]) -> None:
        self.classes = classes
        self.classes_counts = {}
        self.n = 0
        self.correctly_classified = 0
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.n = 0
        self.n_correct = 0
        self.classes_counts = { label: {"n_correct":0, "n":0} for label, c in enumerate(self.classes)}

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        self.n += len(target)
        prediction = torch.argmax(prediction, 1, keepdim=True)
        is_correct = prediction.view(-1) == target
        self.n_correct += sum(is_correct).item()
        for c in self.classes_counts.keys():
            indx = target == c
            self.classes_counts[c]["n_correct"] += sum(is_correct[indx])
            self.classes_counts[c]["n"] += len(is_correct[indx])

        

    def __str__(self):
        '''
        Return a string representation of the performance, accuracy and per class accuracy.
        '''

        accuracy = round(self.accuracy(),4)
        per_class_accuracy = round(self.per_class_accuracy(),4)

        return f"Accuracy: {accuracy}, Class Accuracy: {per_class_accuracy}"


    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        return 0.0 if self.n == 0 else  self.n_correct / self.n
    
    def per_class_accuracy(self) -> float:
        '''
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        return mean([ torch.divide(stats["n_correct"], stats["n"]).nan_to_num().item() for stats in self.classes_counts.values()])
       