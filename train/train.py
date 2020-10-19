import argparse
import glob
import os
import torch

import env
from AudioOnlyDataset import AudioOnlyDataset
from NoiseNN import NoiseReductionNN
from torch.utils import data


def get_dataset(dataset_name):
    dataset = glob.glob(os.path.join(os.environ['DATASET_DIR'] + "/" + dataset_name + "/info", "*.csv"))

    if len(dataset) < env.TRAIN + env.EVALUATION:
        print("ERROR: dataset size is small. len(dataset) =", len(dataset))

    all_nums = range(len(dataset))
    train = all_nums[:env.TRAIN]
    test = all_nums[env.TRAIN:env.TRAIN + env.EVALUATION]

    return train, test


def train_model(model, device, train_loader, criterion, optimizer, scheduler, epoch):
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


def main(args):
    # if not os.path.exists(os.environ['MODEL_DIR']):
    #     os.makedirs(os.environ['MODEL_DIR'])

    env.print_settings()
    print('Using GPUs:', args.gpu0, args.gpu1, args.gpu2, args.gpu3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading model and dataset...")
    trained = None

    model = NoiseReductionNN().to(device)
    train, test = get_dataset("clean_noise")
    train_data = AudioOnlyDataset(train)
    val_data = AudioOnlyDataset(test)

    optimizer = torch.optim.Adam(model.parameters(), lr=3 * 1e-5)
    criterion = torch.nn.CTCLoss(blank=28).to(device)

    train_loader = data.DataLoader(dataset=train_data, batch_size=env.BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_data, batch_size=env.BATCH_SIZE, shuffle=False)

    print("setting trainer...")

    epochs = 10

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4,
                                                    steps_per_epoch=int(len(train_loader)),
                                                    epochs=epochs,
                                                    anneal_strategy='linear')

    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()
        train_model(model, device, train_loader, criterion, optimizer, scheduler, epoch)

    print("saving model...")
    print("done!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu0", "-g0", type=int, default=0, help="Main GPU")
    parser.add_argument("--gpu1", "-g1", type=int, default=-1, help="Second GPU")
    parser.add_argument("--gpu2", "-g2", type=int, default=-1, help="Third GPU")
    parser.add_argument("--gpu3", "-g3", type=int, default=-1, help="Fourth GPU")
    parser.add_argument("--resume", default="", help="Resume the training from snapshot")
    args = parser.parse_args()

    main(args)
