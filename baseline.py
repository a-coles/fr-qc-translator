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
