# -*- coding: utf-8 -*-
# Assemble individual corpora into one big corpus.


import glob
import io
import json
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import train_test_split


class Vocab():
    def __init__(self):
        self.word2idx = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'UNK': 3}
        self.idx2word = {0: 'PAD', 1: 'BOS', 2: 'EOS', 3: 'UNK'}
        self.counts = {}
        self.num_words = 4

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
        if word not in self.counts:
            self.counts[word] = 1
        else:
            self.counts[word] += 1

    def filter_unk(self, threshold=3):
        # If a word occurs under the threshold # of times,
        # remove from the vocab. Dataloader will cast it to UNK.
        infrequent_words = [k for k, v in self.counts.items() if v < threshold]
        word2idx = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'UNK': 3}
        idx2word = {0: 'PAD', 1: 'BOS', 2: 'EOS', 3: 'UNK'}
        for k, v in self.word2idx.items():
            if k in ['PAD', 'BOS', 'EOS', 'UNK']:
                continue
            if k not in infrequent_words:
                word2idx[k] = len(word2idx)
                idx2word[word2idx[k]] = k
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.num_words -= len(infrequent_words)

    def save(self, save_path):
        v = {'word2idx': self.word2idx,
             'idx2word': self.idx2word,
             'num_words': self.num_words}
        with io.open(save_path, mode='w', encoding='utf-8') as fp:
            json.dump(v, fp, indent=4, sort_keys=True, ensure_ascii=False)

    def load(self, load_path):
        with io.open(load_path, mode='r', encoding='utf-8') as fp:
            v = json.load(fp)
        self.word2idx = v['word2idx']
        self.idx2word = v['idx2word']
        self.num_words = v['num_words']

    def get_sentence(self, idx_batch):
        sentences = []
        for idx_tensor in idx_batch:
            idx_np = idx_tensor.numpy()
            sentence = [self.idx2word[str(idx)] for idx in idx_np]
            sentences.append(sentence)
        return sentences

    def get_indices(self, words):
        indices = []
        for word in words:
            try:
                indices.append(self.word2idx[word])
            except KeyError:
                indices.append(self.word2idx['UNK'])
        return indices


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
    for ep in eps_qc:
        with io.open(ep, mode='r', encoding='utf-8') as fp:
            lines = [line.strip() for line in fp.readlines()]
            print('len qc', len(lines))
        simpsons_qc.extend(lines)
    eps_fr = sorted(glob.glob('corpus/simpsons/*_fr_preproc.txt'))
    print('--')

    for ep in eps_fr:
        with io.open(ep, mode='r', encoding='utf-8') as fp:
            lines = [line.strip() for line in fp.readlines()]
            print('len fr', len(lines))
        simpsons_fr.extend(lines)
    simpsons = list(zip(simpsons_qc, simpsons_fr))
    print(simpsons[0])

    # Read in bible corpus
    with io.open('corpus/bible/marc_qc_preproc.txt', mode='r', encoding='utf-8') as fp:
        bible_qc = [line.strip() for line in fp.readlines()]
    with io.open('corpus/bible/marc_fr_preproc.txt', mode='r', encoding='utf-8') as fp:
        bible_fr = [line.strip() for line in fp.readlines()]
    bible = list(zip(bible_qc, bible_fr))
    print(bible[0])

    # Read in querelle corpus
    with io.open('corpus/querelle/querelle_qc_preproc.txt', mode='r', encoding='utf-8') as fp:
        querelle_qc = [line.strip() for line in fp.readlines()]
    with io.open('corpus/querelle/querelle_fr_preproc.txt', mode='r', encoding='utf-8') as fp:
        querelle_fr = [line.strip() for line in fp.readlines()]
    querelle = list(zip(querelle_qc, querelle_fr))
    # If sentences are same, filter them out. NB: This did not help.
    # querelle = [q for q in querelle if q[0] != q[1]]
    print(querelle[-1])

    # Compose list of all examples from all corpora
    examples = simpsons + bible + querelle

    # Heuristic: If the lengths of qc vs fr are very different,
    # there is probably misalignment or noise in the original data.
    # Filter out pairs with really mismatching numbers of tokens.
    # NB: This also did not help.
    # examples = [ex for ex in examples if abs(len(ex[0].split()) - len(ex[1].split())) < 10]
    
    # Clean up other things
    examples = [(ex[0].replace('…', '...'), ex[1].replace('…', '...')) for ex in examples]
    examples = [(ex[0].replace('œ', 'oe'), ex[1].replace('œ', 'oe')) for ex in examples]
    examples = [(ex[0].replace('«', '\"'), ex[1].replace('«', '\"')) for ex in examples]
    examples = [(ex[0].replace('»', '\"'), ex[1].replace('»', '\"')) for ex in examples]
    examples = [(ex[0].replace('’', '\''), ex[1].replace('’', '\'')) for ex in examples]
    examples = [(ex[0].replace('“', '\''), ex[1].replace('“', '\'')) for ex in examples]
    examples = [(ex[0].replace('”', '\''), ex[1].replace('”', '\'')) for ex in examples]
    examples = [(ex[0].replace('\"', '\''), ex[1].replace('\"', '\'')) for ex in examples]

    # Experiment: only take examples under a certain length
    # NB: This helped!
    examples = [ex for ex in examples if len(ex[0].split()) < 25]

    # Shuffle and split into train, valid, test
    train_examples, test_examples = train_test_split(examples,
                                                     test_size=0.1, random_state=42)
    train_examples, valid_examples = train_test_split(train_examples,
                                                      test_size=0.11, random_state=42)

    # Build vocab from training examples
    vocab = Vocab()
    for ex in train_examples:
        vocab.add_example(ex)
    vocab.filter_unk()
    vocab.save('corpus/vocab.json')
    print('Vocab size:', vocab.num_words)

    # Write examples to files
    print('Number of training examples:', len(train_examples))
    print('Number of valid examples:', len(valid_examples))
    print('Number of test examples:', len(test_examples))
    write_examples('corpus/train_qc.txt', 'corpus/train_fr.txt', train_examples)
    write_examples('corpus/valid_qc.txt', 'corpus/valid_fr.txt', valid_examples)
    write_examples('corpus/test_qc.txt', 'corpus/test_fr.txt', test_examples)

    # Histogram of source (qc) example lengths
    print('Number of total examples:', len(examples))
    lens = [len(ex[0].split()) for ex in examples]
    plt.hist(lens, bins=20)
    plt.xticks(range(0, 150, 10))  # NB: change to fit
    plt.axis([0, 150, 0, 2500])    # NB: change to fit
    plt.xlabel('Sequence length (in tokens)')
    plt.ylabel('Number of sequences')
    plt.savefig(os.path.join('corpus', 'histogram.png'))
