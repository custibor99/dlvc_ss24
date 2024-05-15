import torch
from torchvision.models.segmentation import fcn_resnet50


class Resnet50Wrapper(torch.nn.Module):
    def __init__(self, num_classes = 3, pre_trained=True):
        super(Resnet50Wrapper, self).__init__()
        weights = "DEFAULT" if pre_trained else None 
        self.model = fcn_resnet50(weights=weights)
        self.model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        res = self.model(x)
        return res["out"]
