# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa

from allennlp.training.metrics import BLEU  # noqa


class Trans():
    def __init__(self, vocab, cfg, device, name=None, bi=True, att=True, teach_forc_ratio=0.5, patience=3):
        self.device = device
        self.model = TransArch(vocab, cfg, device).to(device)
        self.vocab = vocab
        self.name = name if name else 'seq2seq'

        # Evaluation metrics
        self.bleu = BLEU(exclude_indices=set([0]))  # Exclude padding

        # Logging variables
        self.train_losses, self.valid_losses, self.test_losses = [], [], []
        self.train_bleu, self.valid_bleu, self.test_bleu = [], [], []
        self.patience = patience  # For early stopping
        self.outputs = []

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def log_outputs(self, log_dir):
        '''
        Log the test outputs to a txt.
        '''
        with open(os.path.join(log_dir, '{0}_outputs.txt'.format(self.name)), 'w') as fp:
            for line in self.outputs:
                fp.write('{0}\n'.format(line))

    def log_learning_curves(self, log_dir, graph=True):
        '''
        Logs the learning curve info to a csv.
        '''
        header = 'epoch,train_loss,valid_loss'
        num_epochs = len(self.train_losses)
        with open(os.path.join(log_dir, '{0}_learning_curves.csv'.format(self.name)), 'w') as fp:
            fp.write('{0}\n'.format(header))
            for e in range(num_epochs):
                fp.write('{0},{1},{2}\n'.format(e, self.train_losses[e], self.valid_losses[e]))
        if graph:
            plt.plot(list(range(num_epochs)), self.train_losses, color='blue', label='Train')
            plt.plot(list(range(num_epochs)), self.valid_losses, color='red', label='Valid')
            plt.title('Cross-entropy loss over training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(log_dir, '{0}_learning_curves.png'.format(self.name)))
            plt.clf()

    def log_metrics(self, log_dir, graph=True):
        '''
        Logs evaluation metrics (BLEU, etc.) to a csv.
        '''
        header = 'epoch,train_bleu,valid_bleu'
        num_epochs = len(self.train_bleu)
        with open(os.path.join(log_dir, '{0}_metrics.csv'.format(self.name)), 'w') as fp:
            fp.write('{0}\n'.format(header))
            for e in range(num_epochs):
                fp.write('{0},{1},{2}\n'.format(e, self.train_bleu[e], self.valid_bleu[e]))
        if graph:
            plt.plot(list(range(num_epochs)), self.train_bleu, color='blue', label='Train')
            plt.plot(list(range(num_epochs)), self.valid_bleu, color='red', label='Valid')
            plt.title('BLEU score over training')
            plt.xlabel('Epoch')
            plt.ylabel('BLEU')
            plt.legend()
            plt.savefig(os.path.join(log_dir, '{0}_metrics.png'.format(self.name)))
            plt.clf()

    def generate(self, qc_sentence):
        print('inp qc:', qc_sentence)
        qc_words = ['BOS'] + qc_sentence.split() + ['EOS']
        qc_idx = torch.tensor(self.vocab.get_indices(qc_words)).to(self.device).unsqueeze(0)
        qc_len = torch.tensor(qc_idx.size()[1]).int().cpu().unsqueeze(0)
        pass

    def test(self, test_loader, loss_fn, test_bsz):
        test_loss, test_bleu = self.valid_epoch(test_loader, loss_fn, test_bsz, write_outputs=True)
        print('Test loss: {0}'.format(test_loss))
        print('Test bleu: {0}'.format(test_bleu))
        self.test_losses.append(test_loss)
        self.test_bleu.append(test_bleu)

    def train(self, train_loader, valid_loader, loss_fn=None, lr=1e-2, train_bsz=1, valid_bsz=1, num_epochs=1):
        opt = torch.optim.Adam(self.model.transformer.parameters(), lr=lr)
        for epoch in range(num_epochs):
            train_loss, train_bleu = self.train_epoch(train_loader, loss_fn, opt, train_bsz)
            print('EPOCH {0} \t train_loss {1} \t train_bleu {2}'.format(epoch, train_loss, train_bleu))
            valid_loss, valid_bleu = self.valid_epoch(valid_loader, loss_fn, valid_bsz)
            print('\t valid_loss {1} \t valid_bleu {2}'.format(epoch, valid_loss, valid_bleu))
            # Early stop
            last_val_losses = self.valid_losses[-self.patience:]
            if epoch > self.patience:
                stop = True
                for l in last_val_losses:
                    if valid_loss < l:
                        stop = False
                        break
                if stop:
                    print('Early stopping: validation loss has not improved in {0} epochs.'.format(self.patience))
                    break
            # Log losses
            self.train_losses.append(train_loss)
            self.train_bleu.append(train_bleu)
            self.valid_losses.append(valid_loss)
            self.valid_bleu.append(valid_bleu)

    def train_epoch(self, train_loader, loss_fn, opt, train_bsz=1):
        self.model.train()
        for i, (x, y, x_len, y_len) in enumerate(train_loader):
            opt.zero_grad()
            inp_qc = self.vocab.get_sentence(x)
            print('---')
            print('TRAIN inp qc', ' '.join(inp_qc[0]).replace('PAD', '').strip())
            inp_fr = self.vocab.get_sentence(y)
            print('TRAIN inp_fr', ' '.join(inp_fr[0]).replace('PAD', '').strip())

            x, y = x.to(self.device), y.to(self.device)
            print('src', x.size())
            print('tgt', x.size())
            out = self.model(x, y)

    def valid_epoch(self, valid_loader, loss_fn, valid_bsz=1, write_outputs=False):
        pass


class TransArch(nn.Module):
    def __init__(self, vocab, cfg, device):
        super(TransArch, self).__init__()
        self.embedding = nn.Embedding(vocab.num_words, cfg['embedding_size'])
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12, d_model=['embedding_size'])

    def forward(self, x, y):
        x = self.embedding(x.long()).permute(1, 0, 2)
        print('emb', x.size())
        y = self.embedding(y.long()).permute(1, 0, 2)
        x = self.transformer(x, y)
        return x
