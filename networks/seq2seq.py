# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.training.metrics import BLEU


class Seq2Seq():
    def __init__(self, vocab, cfg, device, name=None, bi=True, att=True):
        self.device = device
        self.model = Seq2SeqArch(vocab, cfg, device, bi=bi, att=att)
        # self.encoder = Encoder(vocab.num_words, cfg['embedding_size'], cfg['hidden_size'], device).to(device)
        # self.decoder = Decoder(cfg['hidden_size'], cfg['embedding_size'], vocab.num_words, device).to(device)
        self.vocab = vocab
        self.name = name if name else 'seq2seq'

        # Evaluation metrics
        self.bleu = BLEU()

        # Logging variables
        self.train_losses, self.valid_losses = [], []
        self.train_bleu, self.valid_bleu = [], []

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

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
        # Whole sequence through encoder
        outputs = torch.zeros(100, 1, self.vocab.num_words).to(self.device)
        enc_hid = self.model.encoder.init_hidden(1).to(self.device)
        enc_out, enc_hid = self.model.encoder(qc_idx, qc_len, enc_hid)
        dec_inp = torch.ones(1, device=self.device) * 1
        dec_hid = enc_hid  # First decoder hidden state is last encoder hidden state

        # One token at a time from decoder
        tok, i = 0, 0
        while tok != 2:  # EOS
            dec_out, dec_hid = self.model.decoder(dec_inp, dec_hid)
            outputs[i] = dec_out
            tok = dec_out.argmax(1)
            dec_inp = tok
            i = i + 1
        pred_tok = torch.argmax(outputs.detach(), dim=2)
        pred_tok = torch.transpose(pred_tok, 1, 0)
        pred_sents = [' '.join(x) for x in self.vocab.get_sentence(pred_tok.cpu())][0]
        pred_sents = pred_sents.replace('PAD', '').strip()
        pred_sents = pred_sents[4:-4]
        print('out fr:', pred_sents)


    def train(self, train_loader, valid_loader, loss_fn=None, lr=1e-2, train_bsz=1, valid_bsz=1, num_epochs=1):
        enc_opt = torch.optim.Adam(self.model.encoder.parameters(), lr=lr)
        dec_opt = torch.optim.Adam(self.model.decoder.parameters(), lr=lr)
        for epoch in range(num_epochs):
            train_loss, train_bleu = self.train_epoch(train_loader, loss_fn, enc_opt, dec_opt, train_bsz)
            print('EPOCH {0} \t train_loss {1} \t train_bleu {2}'.format(epoch, train_loss, train_bleu))
            self.train_losses.append(train_loss)
            self.train_bleu.append(train_bleu)
            valid_loss, valid_bleu = self.valid_epoch(valid_loader, loss_fn, valid_bsz)
            print('\t valid_loss {1} \t valid_bleu {2}'.format(epoch, valid_loss, valid_bleu))
            self.valid_losses.append(valid_loss)
            self.valid_bleu.append(valid_bleu)

    def train_epoch(self, train_loader, loss_fn, enc_opt, dec_opt, train_bsz=1):
        self.model.encoder.train()
        self.model.decoder.train()
        loss_epoch, bleu_epoch = 0.0, 0.0
        for i, (x, y, x_len, y_len) in enumerate(train_loader):
            if i > 10:
                break
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            inp_qc = self.vocab.get_sentence(x)
            print('---')
            print('inp qc', ' '.join(inp_qc[0]))
            inp_fr = self.vocab.get_sentence(y)
            print('inp_fr', ' '.join(inp_fr[0]))

            x, y = x.to(self.device), y.to(self.device)
            enc_hid = self.model.encoder.init_hidden(train_bsz).to(self.device)

            tgt_len = y.size(1)
            loss = 0.0

            # To store decoder outputs
            outputs = torch.zeros(tgt_len, train_bsz, self.vocab.num_words).to(self.device)

            # Whole sequence through encoder
            enc_out, enc_hid = self.model.encoder(x, x_len, enc_hid)

            # First input to the decoder is BOS (hardcoded: idx is 1)
            dec_inp = torch.ones(train_bsz, device=self.device) * 1
            dec_hid = enc_hid  # First decoder hidden state is last encoder hidden state

            # One token at a time from decoder
            for di in range(tgt_len):  # Minus 2 so that we start at 0, and we exclude BOS
                if self.model.decoder.att:
                    dec_out, dec_hid, dec_attn = self.model.decoder(dec_inp, dec_hid, enc_out=enc_out)
                    print('!')
                else:
                    dec_out, dec_hid = self.model.decoder(dec_inp, dec_hid)
                outputs[di] = dec_out
                tok = dec_out.argmax(1)
                # No teacher forcing: next input is current output
                dec_inp = tok

            # Replace the predicted tokens that should be padding with padding
            mask = torch.arange(tgt_len).expand(len(y_len), tgt_len) < y_len.unsqueeze(1)
            mask = torch.transpose(mask, 1, 0).unsqueeze(2).float().to(self.device)
            outputs = outputs * mask

            # When calculating loss, collapse batches together and
            # remove leading BOS (since we feed this to everything)
            all_outputs = outputs.view(-1, outputs.shape[-1])
            all_y = torch.transpose(y, 1, 0)
            all_y = all_y.reshape(-1)
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

    def valid_epoch(self, valid_loader, loss_fn, valid_bsz=1):
        self.model.encoder.eval()
        self.model.decoder.eval()
        loss_epoch, bleu_epoch = 0.0, 0.0
        for i, (x, y, x_len, y_len) in enumerate(valid_loader):
            x, y = x.to(self.device), y.to(self.device)
            enc_hid = self.model.encoder.init_hidden(valid_bsz).to(self.device)

            tgt_len = y.size(1)
            loss = 0.0

            # To store decoder outputs
            outputs = torch.zeros(tgt_len, valid_bsz, self.vocab.num_words).to(self.device)

            # Whole sequence through encoder
            enc_out, enc_hid = self.model.encoder(x, x_len, enc_hid)

            # First input to the decoder is BOS (hardcoded: idx is 1)
            dec_inp = torch.ones(valid_bsz, device=self.device) * 1
            dec_hid = enc_hid  # First decoder hidden state is last encoder hidden state

            # TODO retrun dec attention output's ?
            dec_attns = torch.zeros(100, 100)

            # One token at a time from decoder
            for di in range(tgt_len - 1):
                if self.model.att:
                    dec_out, dec_hid, dec_attn = self.model.decoder(dec_inp, dec_hid, enc_out=enc_out)
                    dec_attns[di] = dec_attn
                else:
                    dec_out, dec_hid = self.model.decoder(dec_inp, dec_hid)
                outputs[di] = dec_out
                tok = dec_out.argmax(1)
                # No teacher forcing: next input is current output
                dec_inp = tok

            # Replace the predicted tokens that should be padding with padding
            mask = torch.arange(tgt_len).expand(len(y_len), tgt_len) < y_len.unsqueeze(1)
            mask = torch.transpose(mask, 1, 0).unsqueeze(2).float().to(self.device)
            outputs = outputs * mask

            # When calculating loss, collapse batches together and
            # remove leading BOS (since we feed this to everything)
            all_outputs = outputs.view(-1, outputs.shape[-1])
            all_y = torch.transpose(y, 1, 0)
            all_y = all_y.reshape(-1)
            loss = loss_fn(all_outputs, all_y.long())

            # Report loss
            loss_batch = loss.item() / tgt_len
            loss_epoch += loss_batch

            # Report BLEU
            pred_tok = torch.argmax(outputs.detach(), dim=2)
            pred_tok = torch.transpose(pred_tok, 1, 0)
            self.bleu(pred_tok, y)
            bleu_batch = self.bleu.get_metric()['BLEU']
            bleu_epoch += bleu_batch

        # Average BLEU over all batches in epoch
        bleu_epoch = bleu_epoch / len(valid_loader)
        return loss_epoch, bleu_epoch


class Seq2SeqArch(nn.Module):
    def __init__(self, vocab, cfg, device, bi=True, att=False):
        '''
        This is a helper class that itself does nothing,
        but putting all the model parts together here facilitates
        saving/loading weights in just one model file.
        '''
        super(Seq2SeqArch, self).__init__()
        self.encoder = Encoder(vocab.num_words, cfg['embedding_size'],
                               cfg['hidden_size'], device,
                               bi=bi).to(device)
        self.decoder = Decoder(cfg['hidden_size'], cfg['embedding_size'],
                               vocab.num_words, device,
                               att=att).to(device)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, device, bi=True):
        super(Encoder, self).__init__()
        self.device = device
        self.bi = bi
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=bi)
        if bi:
            self.fc = nn.Linear(hidden_size * 2, hidden_size)

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
        if self.bi:
            # Combine forward and backward RNN states
            hidden = self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
            hidden = torch.tanh(hidden).unsqueeze(0)
        return output, hidden

    def init_hidden(self, batch_size):
        dim = 2 if self.bi else 1
        return torch.zeros(dim, batch_size, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, device, att=False, max_length=100):
        super(Decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.att = att
        if att:
            self.attn_l1 = nn.Linear(2*self.hidden_size, max_length) # dim=1 ?
            self.attn_l2 = nn.Linear(2*self.hidden_size, self.hidden_size)


    def forward(self, x, hidden, enc_out=None):
        # Input here is always one token at a time,
        # so need to do some unsqueezing to account for length dimension (1)
        x = x.long().unsqueeze(0)
        embedded = self.embedding(x)
        if self.att:
            concat = torch.cat((hidden.squeeze(0), embedded.squeeze(0)), 1)
            att_l1 = self.att_l1(concat)
            attn_weights = F.softmax(att_l1, dim=1)

            # print('enc_out shape', enc_out.shape, 'attn weights unsqueeze', attn_weights.unsqueeze(0).shape)
            # import pdb; pdb.set_trace()
            # batch matrix-matrix product
            context = torch.bmm(attn_weights.unsqueeze(0), enc_out)
            to_gru = torch.cat((embedded.squeeze(0), context.squeeze(0)))
            to_gru = self.attn_comb(to_gru).unsqueeze(0)
        else:
            to_gru = embedded
        to_gru = F.relu(to_gru)
        output, hidden = self.gru(embedded, hidden)
        # TODO output = self.softmax(self.out(output.squeeze(0)), dim=1)
        output = self.out(output.squeeze(0))
        if self.att:
            return output, hidden, attn_weights
        else:
            return output, hidden
