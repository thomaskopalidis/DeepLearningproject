import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # self.relu is a ReLU (Rectified Linear Unit) activation function
        # It's used to introduce non-linearity to the model, allowing it to learn more complex functions
        self.relu = nn.ReLU()
        # self.pool is an average pooling layer with a 2x2 window and a stride of 2.
        # Pooling layers are used to reduce the spatial dimensions (width and height) of the input volume.
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        # self.conv1, self.conv2, self.conv3 are convolutional layers with different numbers of input and output channels, a 5x5 kernel size, a stride of 1, and no padding.
        # These layers are responsible for extracting features from the input images.
        # self.linear1 and self.linear2 are fully connected (linear) layers that map the learned features into the final output.
        # self.linear1 maps from 120 to 84 nodes, and self.linear2 maps from 84 to 10 nodes (typically representing 10 classes for classification).
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

x = torch.randn(64, 1, 32, 32)
print(x)
model = LeNet()
print(model(x).shape)