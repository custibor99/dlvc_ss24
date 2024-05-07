from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch


def convolution_block(fan_in: int, fa_out: int, kernel: int, padding, stride: int):
    block = nn.Sequential(
        nn.Conv2d(fan_in, fa_out, kernel, stride, padding),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return block


def normalized_convolution_block(fan_in: int, fa_out: int, kernel: int, padding, stride: int):
    block = nn.Sequential(
        nn.Conv2d(fan_in, fa_out, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(fa_out),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return block

    
class SimpleCNNTune(nn.Module):
    def __init__(self, conv_filters, fc_neurons, activation_func, dropout_rate):
        super(SimpleCNNTune, self).__init__()
        self.activation = activation_func
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()
        for i in range(len(conv_filters) - 1):
            self.convs.append(nn.Conv2d(conv_filters[i], conv_filters[i+1], 5))

        self.pool = nn.MaxPool2d(2, 2)

        self.fcs = nn.ModuleList()
        for i in range(len(fc_neurons) - 1):
            self.fcs.append(nn.Linear(fc_neurons[i], fc_neurons[i+1]))

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.activation(x)
            x = self.pool(x)

        x = torch.flatten(x, 1)

        for i in range(len(self.fcs) - 1):
            x = self.fcs[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.fcs[-1](x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class DeepCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = convolution_block(3, 16, 5, 2, 1)
        self.conv2 = convolution_block(16, 32, 3, 1, 1)
        self.conv3 = convolution_block(32, 64, 3, 1, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DeepNormalizedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = normalized_convolution_block(3, 16, 5, 2, 1)
        self.conv2 = normalized_convolution_block(16, 32, 3, 1, 1)
        self.conv3 = normalized_convolution_block(32, 64, 3, 1, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
