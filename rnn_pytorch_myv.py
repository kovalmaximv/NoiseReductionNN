import os

import librosa
import torch
import torchaudio
import math
import torch.nn.functional as F

import numpy as np
from torch import nn, optim
from torch.utils import data

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
)
valid_audio_transforms = torchaudio.transforms.MelSpectrogram()


def get_noise_from_sound(signal, noise, snr):
    rms_s = math.sqrt(np.mean(signal ** 2))
    # required RMS of noise
    rms_n = math.sqrt(rms_s ** 2 / (pow(10, snr / 10)))

    # current RMS of noise
    rms_n_current = math.sqrt(np.mean(noise ** 2))
    noise = noise * (rms_n / rms_n_current)

    return noise


def data_processing(data, data_type="train"):
    noise = librosa.load('84-121123-0000-noise.flac')[0]
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        # noise_wave = noise
        # if len(noise) > len(waveform):
        #    noise_wave = noise[0:len(waveform)]
        # noise_wave = get_noise_from_sound(waveform, noise_wave, snr=10)
        # dirty_wave = waveform + noise_wave

        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        #label = torch.Tensor(spec)
        labels.append(spec)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(spec.shape[0] // 2)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return spectrograms, labels, input_lengths, label_lengths


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class NoiseReductionNN(nn.Module):
    def __init__(self, rnn_dim, n_rnn_layers, n_feats, stride=2, dropout=0.1):
        super(NoiseReductionNN, self).__init__()

        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
        ])
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                       100. * batch_idx / len(train_loader), loss.item()))


def main(learning_rate=5e-4, batch_size=2, epochs=10,
         train_url="dev-clean", test_url="test-clean"):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=hparams['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing(x, 'train'),
                                   **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(x, 'valid'),
                                  **kwargs)

    model = NoiseReductionNN(
        hparams['rnn_dim'], hparams['n_rnn_layers'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    # criterion = nn.CTCLoss().to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch)


if __name__ == '__main__':
    main()
