import torch.nn as nn

# 3 layer CNN
class Net3(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_features):
        super(Net3, self).__init__()
        
        feat_size = 16

        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1],  kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(2)
        self.fc = nn.Linear(int((feat_size/8)**2*hidden_channels[2]), out_features)

    # Forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.max_pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

