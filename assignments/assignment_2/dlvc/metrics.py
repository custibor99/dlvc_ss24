from abc import ABCMeta, abstractmethod
import torch

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


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes
        self.n_classes = len(classes)
        self.score_sum = 0.0
        self.n_datapoints = 0

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.score_sum = 0.0
        self.n_batches = 0

    def _explode_tensor(self,input, n_classes = 3):
        """
        expects a tensor of shape [b,w,h]
        and returns a tensor of shape [b,w,h,n_classes]
        label ranges should be from 0 to n_classes
        """
        b, w, h = input.shape
        exploded = torch.zeros([b, n_classes, w,h], dtype=torch.uint8)
        for i in range(0, n_classes):
            mask = input.view(-1,w,h) == i
            mask = mask.view([b, w, h])
            exploded[:,i,:,:] = mask            
        return exploded

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

       ##TODO implement
        with torch.no_grad():
            b, c, h, w = prediction.shape
            prediction = prediction.argmax(dim=1).view(b,h,w)
            prediction = self._explode_tensor(prediction)
            target = self._explode_tensor(target.view(b,w,h))

            tp = prediction.logical_and(target).sum(dim=2).sum(dim=2)
            n_labels = prediction.logical_or(target).sum(dim=2).sum(dim=2)
            self.score_sum += torch.mean(torch.nan_to_num(tp / n_labels)).item()
            self.n_batches += 1
            #print(tp, n_labels)

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        score = self.mIoU()
        return f"mIou: {score}"
          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        score = 0.0 if self.n_batches == 0 else self.score_sum / self.n_batches
        return round(score, 2)