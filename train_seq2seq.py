# -*- coding: utf-8 -*-
# Training routines.

import argparse
import json
import torch
import torch.nn as nn

from assemble_corpus import Vocab
from dataset import QcFrDataset, pad_collate
from torch.utils.data import DataLoader

from networks.seq2seq import Seq2Seq


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Training config.')
    args = parser.parse_args()

    # Load config and vocab
    with open(args.config_file, 'r') as fp:
        cfg = json.load(fp)
    vocab = Vocab()
    vocab.load('corpus/vocab.json')

    # Create training dataset
    train_dataset = QcFrDataset('corpus/train_qc.txt', 'corpus/train_fr.txt', 'corpus/vocab.json')
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_bsz'],
                              shuffle=True, drop_last=True, collate_fn=pad_collate)

    valid_dataset = QcFrDataset('corpus/valid_qc.txt', 'corpus/valid_fr.txt', 'corpus/vocab.json')
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['valid_bsz'],
                              shuffle=True, drop_last=True)

    # Training loop
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['PAD'])
    model = Seq2Seq(vocab, cfg, device)
    model.train(train_loader, loss_fn=criterion, train_bsz=cfg['train_bsz'], num_epochs=cfg['num_epochs'])
    model.log_learning_curves(log_dir='log')
    model.log_metrics(log_dir='log')
