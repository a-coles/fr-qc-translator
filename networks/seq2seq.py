# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(Encoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x, x_lens, hidden):
        x = x.long()
        # print('enc inp', x.size())
        embedded = self.embedding(x).unsqueeze(0)
        # Ignore the padding during backprop
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens,
                                                           batch_first=True,
                                                           enforce_sorted=False)
        # print('enc emb', embedded.size())
        output = embedded
        output, hidden = self.gru(output, hidden)
        # print('enc out', output.size())
        # output, output_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(Decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        x = x.long()
        # print('dec inp', x.size())
        output = self.embedding(x).unsqueeze(0)  #.view(1, 1, -1)  # TODO: ?
        # print('dec emb', output.size())
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))  # ?
        return output, hidden

    # def init_hidden(self):
    #     return torch.zeros(1, 1, self.hidden_size)


# class Seq2Seq(nn.Module):
#     def __init__():
#         super(Seq2Seq, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
