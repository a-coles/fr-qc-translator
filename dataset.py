# -*- coding: utf-8 -*-
# Data loading.

import io
import torch
import torch.nn.functional as F

from assemble_corpus import Vocab
from torch.utils.data import Dataset


class QcFrDataset(Dataset):
    def __init__(self, qc_file, fr_file, vocab_file, transform=None):
        with io.open(qc_file, mode='r', encoding='utf-8') as fp:
            qc_lines = [line.strip() for line in fp.readlines()]
        with io.open(fr_file, mode='r', encoding='utf-8') as fp:
            fr_lines = [line.strip() for line in fp.readlines()]
        self.examples = list(zip(qc_lines, fr_lines))
        self.vocab = Vocab()
        self.vocab.load(vocab_file)
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        qc_words = ['BOS'] + example[0].split() + ['EOS']
        fr_words = ['BOS'] + example[1].split() + ['EOS']
        qc_idx = torch.tensor([self.vocab.word2idx[word] for word in qc_words])
        fr_idx = torch.tensor([self.vocab.word2idx[word] for word in fr_words])
        # if self.transform:
        #     sample = self.transform(sample)
        qc_idx, fr_idx = qc_idx.int(), fr_idx.int()
        return (qc_idx, fr_idx)


def pad_tensor(tensor, max_len):
    to_add = max_len - tensor.size(0)
    tensor = F.pad(tensor, pad=(0, to_add), mode='constant', value=0)
    return tensor


def pad_collate(batch):
    # qc = [example[0] for example in batch]
    # fr = [example[1] for example in batch]
    (qc, fr) = zip(*batch)
    qc_lens = [len(x) for x in qc]
    fr_lens = [len(x) for x in fr]
    # Get the max length over all examples in qc + fr
    max_len = 0
    for example in (qc + fr):
        if example.size(0) > max_len:
            max_len = example.size(0)

    # Pad all examples to the max length
    #qc = [pad_tensor(example, max_len) for example in qc]
    #fr = [pad_tensor(example, max_len) for example in fr]
    qc = torch.nn.utils.rnn.pad_sequence(qc, batch_first=True, padding_value=0)
    fr = torch.nn.utils.rnn.pad_sequence(fr, batch_first=True, padding_value=0)

    #qc = torch.stack(qc)
    #fr = torch.stack(fr)
    return qc, fr, qc_lens, fr_lens



