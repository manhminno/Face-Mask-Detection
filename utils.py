import numpy as np
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.autograd import Variable

data_transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

label2id = {
    0: 'No Mask',
    1: 'Mask'
}

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  #16
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   #8
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=1024, out_features=2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 4096)
        out = self.fc(x)

        return out