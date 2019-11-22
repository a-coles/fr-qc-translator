# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.training.metrics import BLEU


class Seq2Seq():
    def __init__(self, vocab, cfg, device):
        self.device = device
        self.encoder = Encoder(vocab.num_words, cfg['embedding_size'], cfg['hidden_size'], device).to(device)
        self.decoder = Decoder(cfg['hidden_size'], cfg['embedding_size'], vocab.num_words, device).to(device)
        self.vocab = vocab
        self.name = 'seq2seq'

        # Evaluation metrics
        self.bleu = BLEU()

        # Logging variables
        self.train_losses, self.valid_losses = [], []
        self.train_bleu, self.valid_bleu = [], []

    def log_learning_curves(self, log_dir, graph=True):
        '''
        Logs the learning curve info to a csv.
        '''
        header = 'epoch,train_loss,valid_loss'
        num_epochs = len(self.train_losses)
        with open(os.path.join(log_dir, '{0}_learning_curves.csv'.format(self.name)), 'w') as fp:
            fp.write('{0}\n'.format(header))
            for e in range(num_epochs):
                fp.write('{0},{1}\n'.format(e, self.train_losses[e])) #, self.valid_losses[e]))
        if graph:
            plt.plot(list(range(num_epochs)), self.train_losses, color='blue', label='Train')
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
        header = 'epoch,train_bleu'
        num_epochs = len(self.train_bleu)
        with open(os.path.join(log_dir, '{0}_metrics.csv'.format(self.name)), 'w') as fp:
            fp.write('{0}\n'.format(header))
            for e in range(num_epochs):
                fp.write('{0},{1}\n'.format(e, self.train_bleu[e]))
        if graph:
            plt.plot(list(range(num_epochs)), self.train_bleu, color='blue', label='Train')
            plt.title('BLEU score over training')
            plt.xlabel('Epoch')
            plt.ylabel('BLEU')
            plt.legend()
            plt.savefig(os.path.join(log_dir, '{0}_metrics.png'.format(self.name)))
            plt.clf()

    def train(self, train_loader, loss_fn=None, train_bsz=1, num_epochs=1):
        enc_opt = torch.optim.Adam(self.encoder.parameters())
        dec_opt = torch.optim.Adam(self.decoder.parameters())
        for epoch in range(num_epochs):
            train_loss, train_bleu = self.train_epoch(train_loader, loss_fn, enc_opt, dec_opt, train_bsz)
            print('EPOCH {0} \t train_loss {1} \t train_bleu {2}'.format(epoch, train_loss, train_bleu))
            self.train_losses.append(train_loss)
            self.train_bleu.append(train_bleu)

    def train_epoch(self, train_loader, loss_fn, enc_opt, dec_opt, train_bsz=1):
        self.encoder.train()
        self.decoder.train()
        loss_epoch, bleu_epoch = 0.0, 0.0
        for i, (x, y, x_len, y_len) in enumerate(train_loader):
            if i > 10:
                break
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            inp_qc = self.vocab.get_sentence(x)
            print('inp qc', ' '.join(inp_qc[0]))
            inp_fr = self.vocab.get_sentence(y)
            print('inp_fr', ' '.join(inp_fr[0]))

            x, y = x.to(self.device), y.to(self.device)
            enc_hid = self.encoder.init_hidden(train_bsz).to(self.device)

            tgt_len = y.size(1)
            loss = 0.0

            # To store decoder outputs
            outputs = torch.zeros(tgt_len, train_bsz, self.vocab.num_words).to(self.device)

            # Whole sequence through encoder
            enc_out, enc_hid = self.encoder(x, x_len, enc_hid)

            # First input to the decoder is BOS (hardcoded: idx is 1)
            dec_inp = torch.ones(train_bsz, device=self.device) * 1
            dec_hid = enc_hid  # First decoder hidden state is last encoder hidden state

            # One token at a time from decoder
            for di in range(1, tgt_len):
                dec_out, dec_hid = self.decoder(dec_inp, dec_hid)
                outputs[di] = dec_out
                tok = dec_out.argmax(1)
                # No teacher forcing: next input is current output
                dec_inp = tok

            # When calculating loss, collapse batches together and
            # remove leading BOS (since we feed this to everything)
            all_outputs = outputs[1:].view(-1, outputs.shape[-1])
            all_y = torch.transpose(y, 1, 0)
            all_y = all_y[1:].reshape(-1)
            loss = loss_fn(all_outputs, all_y.long())

            loss.backward()
            pred_tok = torch.argmax(outputs.detach(), dim=2)
            pred_tok = torch.transpose(pred_tok, 1, 0)
            pred_sents = [' '.join(x) for x in self.vocab.get_sentence(pred_tok.cpu())]
            print(pred_sents[0])

            enc_opt.step()
            dec_opt.step()

            # Report loss
            loss_batch = loss.item() / tgt_len
            loss_epoch += loss_batch

            # Report BLEU
            self.bleu(pred_tok, y)
            bleu_batch = self.bleu.get_metric()['BLEU']
            bleu_epoch += bleu_batch

        # Average BLEU over all batches in epoch
        bleu_epoch = bleu_epoch / len(train_loader)
        return loss_epoch, bleu_epoch


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, device):
        super(Encoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, x, x_lens, hidden):
        x = x.long()
        embedded = self.embedding(x)
        # Ignore the padding through the RNN
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens,
                                                           batch_first=True,
                                                           enforce_sorted=False)
        output, hidden = self.gru(embedded, hidden)
        # Re-pad
        output, output_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, device):
        super(Decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        # Input here is always one token at a time,
        # so need to do some unsqueezing to account for length dimension (1)
        x = x.long().unsqueeze(0)
        embedded = self.embedding(x)
        embedded = F.relu(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output.squeeze(0))
        return output, hidden
