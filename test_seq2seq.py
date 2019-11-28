# -*- coding: utf-8 -*-
# Testing routines.

import argparse
import json
import os
import torch
import torch.nn as nn

from assemble_corpus import Vocab
from dataset import QcFrDataset, pad_collate
from networks.seq2seq import Seq2Seq
from torch.utils.data import DataLoader


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Testing config. Should be same as used to train.')
    parser.add_argument('model_path', type=str, help='Path to model to test with.')
    parser.add_argument('--log_dir', type=str, default='log', help='Log output dir.')
    parser.add_argument('--name', type=str, default=None, help='Name for model.')
    parser.add_argument('--bi', action='store_true', help='Use a bidirectional encoder.')
    parser.add_argument('--att', action='store_true', help='Use a decoder with attention.')
    args = parser.parse_args()

    # Load config and vocab
    with open(args.config_file, 'r') as fp:
        cfg = json.load(fp)
    vocab = Vocab()
    vocab.load('corpus/vocab.json')

    # Prepare output dirs
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # Create test dataset
    test_dataset = QcFrDataset('corpus/test_qc.txt', 'corpus/test_fr.txt', 'corpus/vocab.json')
    test_loader = DataLoader(test_dataset, batch_size=cfg['test_bsz'],
                             shuffle=True, drop_last=True, collate_fn=pad_collate)

    # Perform test
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['PAD'])
    model = Seq2Seq(vocab, cfg, device,
                    name=args.name, bi=args.bi, att=args.att, teach_forc_ratio=cfg['teacher_forcing_ratio'],
                    patience=cfg['patience'])
    model.load_model(args.model_path)
    model.test(test_loader, criterion, cfg['test_bsz'])
    model.log_learning_curves(log_dir=args.log_dir, graph=False)
    model.log_metrics(log_dir=args.log_dir, graph=False)
    model.log_outputs(log_dir=args.log_dir)
