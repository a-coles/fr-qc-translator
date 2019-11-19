# -*- coding: utf-8 -*-
# Assemble individual corpora into one big corpus.


import glob
import io
import json
import pickle
import os

from sklearn.model_selection import train_test_split


class Vocab():
    def __init__(self):
        self.word2idx = {'PAD': 0, 'BOS': 1, 'EOS': 2}
        # Always start with the known beginning and ending tokens, and padding
        self.idx2word = {0: 'PAD', 1: 'BOS', 2: 'EOS'}
        self.num_words = 3

    def add_example(self, example):
        qc_example = example[0]
        fr_example = example[1]
        for word in qc_example.split():
            self.add_word(word)
        for word in fr_example.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_words
            self.idx2word[self.num_words] = word
            self.num_words += 1

    def save(self, save_path):
        v = {'word2idx': self.word2idx,
             'idx2word': self.idx2word,
             'num_words': self.num_words}
        with io.open(save_path, mode='w', encoding='utf-8') as fp:
            json.dump(v, fp, indent=4, sort_keys=True, ensure_ascii=False)

    def get_sentence(self, idx_batch):
        sentences = []
        for idx_tensor in idx_batch:
            idx_np = idx_tensor.numpy()
            sentence = [self.idx2word[str(idx)] for idx in idx_np]
            sentences.append(sentence)
        return sentences

    def load(self, load_path):
        with io.open(load_path, mode='r', encoding='utf-8') as fp:
            v = json.load(fp)
        self.word2idx = v['word2idx']
        self.idx2word = v['idx2word']
        self.num_words = v['num_words']


def write_examples(qc_file, fr_file, examples):
    with io.open(qc_file, mode='w', encoding='utf-8') as fp1:
        with io.open(fr_file, mode='w', encoding='utf-8') as fp2:
            qc = [ex[0] for ex in examples]
            fr = [ex[1] for ex in examples]
            for x in qc:
                fp1.write('{0}\n'.format(x))
            for x in fr:
                fp2.write('{0}\n'.format(x))


if __name__ == '__main__':
    # Read in simpsons corpus
    simpsons_qc, simpsons_fr = [], []
    eps_qc = sorted(glob.glob('corpus/simpsons/*_qc_preproc.txt'))
    print(eps_qc)
    for ep in eps_qc:
        with io.open(ep, mode='r', encoding='utf-8') as fp:
            lines = [line.strip() for line in fp.readlines()]
        simpsons_qc.extend(lines)
    eps_fr = sorted(glob.glob('corpus/simpsons/*_fr_preproc.txt'))
    print(eps_fr)
    for ep in eps_fr:
        with io.open(ep, mode='r', encoding='utf-8') as fp:
            lines = [line.strip() for line in fp.readlines()]
        simpsons_fr.extend(lines)
    simpsons = list(zip(simpsons_qc, simpsons_fr))
    print(simpsons[0])

    # Compose list of all examples from all corpora
    examples = simpsons

    # Shuffle and split into train, valid, test
    train_examples, test_examples = train_test_split(examples,
                                                     test_size=0.2, random_state=42)
    train_examples, valid_examples = train_test_split(train_examples,
                                                      test_size=0.2, random_state=42)

    # Build vocab from training examples
    vocab = Vocab()
    for ex in train_examples:
        vocab.add_example(ex)
    vocab.save('corpus/vocab.json')

    # Write examples to files
    write_examples('corpus/train_qc.txt', 'corpus/train_fr.txt', train_examples)
    write_examples('corpus/valid_qc.txt', 'corpus/valid_fr.txt', valid_examples)
    write_examples('corpus/test_qc.txt', 'corpus/test_fr.txt', test_examples)
