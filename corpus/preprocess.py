# -*- coding: utf-8 -*-
# Clean individual corpora.

import argparse
import glob
import io
import os
import re
import spacy


def clean_simpsons(in_path):
    with io.open(in_path, mode='r', encoding='utf-8') as fp:
        lines = fp.readlines()
    clean_lines = []
    for line in lines:
        if 'SEQ' in line:
            continue
        line = re.sub(r'\[.*\]', '', line)  # Remove [text in brackets]
        line = re.sub(r'^.*:', '', line)  # Remove Character : at beginning of line
        line = re.sub(r'!+', '!', line)     # !!! -> !
        line = re.sub(r'\?+', '?', line)    # ??? -> ?
        if not line.strip() or line in ['', '\n', '\r\n']:
            continue
        clean_lines.append(line)
    print(clean_lines)
    return clean_lines


def tokenize(lines, nlp):
    tokenized_lines = []
    for line in lines:
        doc = nlp(line.lower())
        tokens = [x.text for x in doc]
        tokenized_lines.append(' '.join(tokens))
    return tokenized_lines


def tokenize_file(in_path, out_path, nlp):
    with io.open(in_path, mode='r', encoding='utf-8') as fp:
        lines = fp.readlines()
    with io.open(out_path, mode='w', encoding='utf-8') as fp:
        tokenized_lines = tokenize(lines)
        for tl in tokenized_lines:
            fp.write(tl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bible', action='store_true')
    parser.add_argument('--simpsons', action='store_true')
    parser.add_argument('--querelle', action='store_true')
    args = parser.parse_args()

    nlp = spacy.load('fr_core_news_sm')

    if args.bible:
        print('Tokenizing marc_qc...')
        tokenize_file('bible/marc_qc.txt', 'bible/marc_qc_preproc.txt', nlp)
        print('Tokenizing marc_fr...')
        tokenize_file('bible/marc_fr.txt', 'bible/marc_fr_preproc.txt', nlp)

    if args.simpsons:
        eps = glob.glob('simpsons/*_qc.txt') + glob.glob('simpsons/*_fr.txt')
        for ep in eps:
            clean_lines = clean_simpsons(ep)
            tokenized_lines = tokenize(clean_lines, nlp)
            name = os.path.join('simpsons', os.path.basename(ep).split('.')[0] + '_preproc.txt')
            with io.open(name, mode='w', encoding='utf-8') as fp:
                for tl in tokenized_lines:
                    fp.write(tl)

    if args.querelle:
        with io.open('querelle/querelle_qc.txt', mode='r', encoding='utf-8') as fp:
            text = ' '.join([line.strip() for line in fp.readlines()])
            text = text.replace('\"', '')
        text = re.split(r'(?<=(\.|\?|!)) ', text)
        text = [line for line in text if len(line) > 1]
        text = tokenize(text, nlp)
        with io.open('querelle/querelle_qc_preproc.txt', mode='w', encoding='utf-8') as fp:
            for sent in text:
                fp.write('{0}\n'.format(sent))
        with io.open('querelle/querelle_fr.txt', mode='r', encoding='utf-8') as fp:
            text = ' '.join([line.strip() for line in fp.readlines()])
            text = text.replace('\"', '')
        text = re.split(r'(?<=(\.|\?|!)) ', text)
        text = [line for line in text if len(line) > 1]
        text = tokenize(text, nlp)
        with io.open('querelle/querelle_fr_preproc.txt', mode='w', encoding='utf-8') as fp:
            for sent in text:
                fp.write('{0}\n'.format(sent))
