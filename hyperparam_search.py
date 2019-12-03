# -*- coding: utf-8 -*-
# Training routines.

import argparse
import json
import os
import torch
import torch.nn as nn

from assemble_corpus import Vocab
from dataset import QcFrDataset, pad_collate
from networks.seq2seq import Seq2Seq
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Training config.')
    parser.add_argument('--log_dir', type=str, default='log', help='Log output dir.')
    parser.add_argument('--model_dir', type=str, default='models', help='Saved model dir.')
    parser.add_argument('--name', type=str, default=None, help='Name for model.')
    parser.add_argument('--continue_model', type=str, help='Path to model for continuing training.')

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
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    param_grid = {'bsz': cfg['train_bsz'],
                  'learning_rate': cfg['learning_rate'],
                'teacher_forcing_ratio': cfg['teacher_forcing_ratio']
    
                  }
    for params in list(ParameterGrid(param_grid)):

        print(params)
        # Create training dataset
        train_dataset = QcFrDataset('corpus/train_qc.txt', 'corpus/train_fr.txt', 'corpus/vocab.json')
        train_loader = DataLoader(train_dataset, batch_size=params['bsz'],
                                  shuffle=True, drop_last=True, collate_fn=pad_collate)

        valid_dataset = QcFrDataset('corpus/valid_qc.txt', 'corpus/valid_fr.txt', 'corpus/vocab.json')
        valid_loader = DataLoader(valid_dataset, batch_size=params['bsz'],
                                  shuffle=True, drop_last=True, collate_fn=pad_collate)

        
        # Training loop
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['PAD'])
        model = Seq2Seq(vocab, cfg, device,
                        name=args.name, bi=args.bi, att=args.att, teach_forc_ratio=params['teacher_forcing_ratio'],
                        patience=cfg['patience'])
        if args.continue_model:
            print('Continuing training from {0}'.format(args.continue_model))
            model.load_model(args.continue_model)

        model.train(train_loader, valid_loader,
                    loss_fn=criterion,
                    lr=params['learning_rate'],
                    train_bsz=params['bsz'],
                    valid_bsz=params['bsz'],
                    num_epochs=cfg['num_epochs'])

        output_dir = os.path.join(args.log_dir, str(params))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # model.log_learning_curves(log_dir=output_dir)
        # model.log_metrics(log_dir=output_dir)
        model.save_model(os.path.join(args.model_dir,'{}.pt'.format(str(params))))


