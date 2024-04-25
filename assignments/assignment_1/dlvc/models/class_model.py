import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''
        if suffix:
            save_dir = save_dir.with_suffix(suffix)
        torch.save(self.net.state_dict(), save_dir)

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        
        self.net = self.net.load_state_dict(path)