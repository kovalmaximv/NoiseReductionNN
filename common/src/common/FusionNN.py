import torch
import torch.nn.functional as F
from torch import nn

import env


class FusionNN(nn.Module):
    def __init__(self):
        super(FusionNN, self).__init__()
        if env.AUDIO_LEN % env.FC_ROW != 0:
            print("invalid fc layer parameter")
            import sys
            sys.exit(1)

        # output size?
        self.lstm = nn.LSTM(num_layers=1, input_size=2056, hidden_size=300, dropout=0.0, bidirectional=True)

        self.fc1 = nn.Linear(in_features=600 * env.FC_ROW, out_features=600 * env.FC_ROW)
        self.fc2 = nn.Linear(in_features=600 * env.FC_ROW, out_features=600 * env.FC_ROW)
        self.fc3 = nn.Linear(in_features=600 * env.FC_ROW, out_features=env.AUDIO_CHANNELS * 257 * env.FC_ROW)

        self.bn1 = nn.BatchNorm2d(600 * env.FC_ROW)
        self.bn2 = nn.BatchNorm2d(600 * env.FC_ROW)
        self.bn3 = nn.BatchNorm2d(env.AUDIO_CHANNELS * 257 * env.FC_ROW)

    def forward(self, x):
        # x = (b, 301, 2568)
        batch_size = x.shape[0]

        # Array to List
        xs = [i for i in x]
        ys = self.lstm(hx=None, cx=None, xs=xs)[2]

        y = torch.stack(ys)
        y = F.leaky_relu(self.bn0(y))

        y = torch.reshape(
            y, shape=(batch_size * int(env.AUDIO_LEN / env.FC_ROW), -1))

        y = F.leaky_relu(self.bn1(self.fc1(y)))
        y = F.leaky_relu(self.bn2(self.fc2(y)))
        y = F.sigmoid(self.bn3(self.fc3(y)))

        y = torch.reshape(
            y, shape=(batch_size, 1, env.AUDIO_CHANNELS, env.AUDIO_LEN, 257))

        return y
