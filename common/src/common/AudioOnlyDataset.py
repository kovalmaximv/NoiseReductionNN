import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset



class AudioOnlyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i: int):
        info = pd.read_csv(
            os.environ['DATASET_DIR'] + "/clean_noise/info/{0}.csv".format(i))

        noise = np.load(info['mix'][0]).T.astype(np.float32)
        clean = np.load(info['clean'][0]).T.astype(np.float32)

        return noise, clean, len(noise), len(clean)
