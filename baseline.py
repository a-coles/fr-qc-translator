import torch
import torch.nn as nn
import torch.optim as optim

# ENCODER
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        # TODO add dropout, relu?
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, layers)

    def forward(self, x):
        out, (hid, cell) = self.lstm(self.embedding(x))
        return hid, cell

    # TODO init hidden state
    # TODO test the encoder
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# DECODER
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers = layers
        # TODO add dropout
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, layers)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, dec_input, hid, cell):
        # accept one token per time step
        dec_input = dec_input.unsqueeze(0)
        out, (hid, cell) = self.lstm(self.embedding(dec_input), (hid, cell))
        preds = self.output(out.squeeze(0))
        return preds, hid, cell

# # seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, x, y):
        # init decoder
        hid, cell = self.encoder(x)
        y_vocab_size = self.decoder.vocab_size
        max_lengh,bsz = y.shape[0],y.shape[1]
        # for storing outputs
        outs = torch.zeros(max_lengh, bsz, y_vocab_size)
        outs = outs.to(self.device)
        # <BOS> token
        dec_input = y[0, :]
        for tok in range(1, max_lengh):
            out, hid, cell = self.decoder(dec_input, hid, cell)
            outs[tok] = out
            # outs = self.softmax(self.output(out))
            outs = output.argmax(1)
        return outs
