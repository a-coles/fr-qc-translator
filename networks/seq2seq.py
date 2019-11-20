# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq():
    def __init__(self, vocab, cfg, device):
        self.device = device
        self.encoder = Encoder(vocab.num_words, cfg['embedding_size'], cfg['hidden_size'], device).to(device)
        self.decoder = Decoder(cfg['hidden_size'], cfg['embedding_size'], vocab.num_words, device).to(device)
        self.vocab = vocab
        self.name = 'seq2seq'

        # Logging variables
        self.train_losses, self.valid_losses = [], []
        self.train_bleu, self.valid_bleu = [], []

    def log_learning_curves(self):
        '''
        Logs the learning curve info to a csv.
        TODO: update with BLEU.
        '''
        header = 'epoch,train_loss,valid_loss'
        num_epochs = len(self.train_losses)
        with open(os.path.join('log', '{0}_learning_curves.log'.format(self.name)), 'w') as fp:
            fp.write('{0}\n'.format(header))
            for e in range(num_epochs):
                fp.write('{0},{1}\n'.format(e, self.train_losses[e]))#, self.valid_losses[e]))

    def train(self, train_loader, loss_fn=None, train_bsz=1, num_epochs=1):
        enc_opt = torch.optim.Adam(self.encoder.parameters())
        dec_opt = torch.optim.Adam(self.decoder.parameters())
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, loss_fn, enc_opt, dec_opt, train_bsz)
            print('EPOCH {0} \t train_loss {1}'.format(epoch, train_loss))
            self.train_losses.append(train_loss)

    def train_epoch(self, train_loader, loss_fn, enc_opt, dec_opt, train_bsz=1):
        self.encoder.train()
        self.decoder.train()
        loss_epoch = 0.0
        for i, (x, y, x_len, y_len) in enumerate(train_loader):
            if i > 100:
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
            loss = 0

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
            all_y = torch.transpose(y[1:].reshape(-1), 1, 0)
            loss = loss_fn(all_outputs, all_y.long())

            loss.backward()
            pred_tok = torch.transpose(torch.argmax(outputs.detach(), dim=2), 1, 0)
            print(' '.join(self.vocab.get_sentence(pred_tok.cpu())[0]))

            enc_opt.step()
            dec_opt.step()
            loss_batch = loss.item() / tgt_len
            loss_epoch += loss_batch
        return loss_epoch


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
        embedded = self.embedding(x)#.permute(0, 2, 1)#.unsqueeze(0)  # TODO: ?
        embedded = F.relu(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output.squeeze(0))
        return output, hidden
