import torch.nn.functional as F
from torch import nn

import env


class AudioNN(nn.Module):
    def __init__(self):
        super(AudioNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=env.AUDIO_CHANNELS, out_channels=96, stride=1,
                               kernel_size=(1, 7), dilation=(1, 1), padding=(0, 3))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, stride=1, kernel_size=(7, 1),
                               dilation=(1, 1), padding=(3, 0))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=96, stride=1, kernel_size=(5, 5),
                               dilation=(2, 1), padding=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=96, stride=1, kernel_size=(5, 5),
                               dilation=(4, 1), padding=(4, 2))
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=8, stride=1, kernel_size=(1, 1),
                               dilation=(1, 1), padding=(0, 0))

        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(96)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(8)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))

        return x