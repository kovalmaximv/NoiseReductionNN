from torch import nn

from AudioNN import AudioNN
from FusionNN import FusionNN


class NoiseReductionNN(nn.Module):
    def __init__(self):
        super(NoiseReductionNN, self).__init__()
        self.audio_nn = AudioNN()
        self.fusion_nn = FusionNN()

    def forward(self, x):
        x = self.audio_nn.forward(x)
        x = self.fusion_nn.forward(x)

        return x
