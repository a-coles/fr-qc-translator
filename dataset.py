# -*- coding: utf-8 -*-
# Data loading.

import io
import torch

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
        return qc_idx, fr_idx



