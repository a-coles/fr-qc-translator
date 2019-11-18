# -*- coding: utf-8 -*-
# Training routines.

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import QcFrDataset
from torch.utils.data import DataLoader

from baseline import Encoder, Decoder, Seq2Seq
from train import train


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

    valid_dataset = QcFrDataset('corpus/valid_qc.txt', 'corpus/valid_fr.txt', 'corpus/vocab.json')
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['valid_bsz'], shuffle=True)

    test_dataset = QcFrDataset('corpus/test_qc.txt', 'corpus/test_fr.txt', 'corpus/vocab.json')
    test_loader = DataLoader(test_dataset, batch_size=cfg['test_bsz'], shuffle=True)

    # initialization
    encoder = Encoder(cfg['vocab_size'], cfg['embedding_size'], cfg['hidden_size'], cfg['layers'])
    decoder = Decoder(cfg['vocab_size'], cfg['embedding_size'], cfg['hidden_size'], cfg['layers'])

    seq2seq = Seq2Seq(encoder, decoder, device)
    model = seq2seq.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0
        # TODO consider teacher forcing, gradient clipping,
        for idx, (qc_idx, fr_idx) in enumerate(train_loader):
            optimizer.zero_grad()
            qc_idx, fr_idx = qc_idx.to(device), fr_idx.to(device)
            out = model(qc_idx, fr_idx)
            # break
        #     loss = criterion(out, qc_idx)
        #     loss.backward()
        #     optimizer.step()
        #     total_loss += loss.item()
        #
        # total_loss/len(train_loader)

