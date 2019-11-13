# -*- coding: utf-8 -*-
# Training routines.

import argparse
import json
import torch

from dataset import QcFrDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Training config.')
    args = parser.parse_args()

    # Load config
    with open(args.config_file, 'r') as fp:
        cfg = json.load(fp)

    # Create training dataset
    train_dataset = QcFrDataset('corpus/train_qc.txt', 'corpus/train_fr.txt', 'corpus/vocab.json')
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_bsz'], shuffle=True)

    # Loop over dataset
    # for i, batch in enumerate(train_loader):
    #     for item in batch:
    #         print(item)
    #     break
    for i in range(len(train_dataset)):
        print(train_dataset[i])
        break
