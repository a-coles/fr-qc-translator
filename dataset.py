# -*- coding: utf-8 -*-
# Data loading.

import io

from torch.utils.data import Dataset


class QcFrDataset(Dataset):
    def __init__(self, qc_file, fr_file, transform=None):
        with io.open(qc_file, mode='r', encoding='utf-8') as fp:
            qc_lines = [line.strip() for line in fp.readlines()]
        with io.open(fr_file, mode='r', encoding='utf-8') as fp:
            fr_lines = [line.strip() for line in fp.readlines()]
        self.examples = list(zip(qc_lines, fr_lines))
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]
        print('GETTING', sample)
        if self.transform:
            sample = self.transform(sample)
        return sample[0], sample[1]
