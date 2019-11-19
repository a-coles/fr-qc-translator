# -*- coding: utf-8 -*-
# Training routines.

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim

from assemble_corpus import Vocab
from dataset import QcFrDataset, pad_collate
from torch.utils.data import DataLoader

#from baseline import Encoder, Decoder, Seq2Seq
from networks.seq2seq import Encoder, Decoder


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Training config.')
    args = parser.parse_args()

    # Load config and vocab
    with open(args.config_file, 'r') as fp:
        cfg = json.load(fp)
    vocab = Vocab()
    vocab.load('corpus/vocab.json')

    # Create training dataset
    train_dataset = QcFrDataset('corpus/train_qc.txt', 'corpus/train_fr.txt', 'corpus/vocab.json')
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_bsz'],
                              shuffle=True, collate_fn=pad_collate)

    valid_dataset = QcFrDataset('corpus/valid_qc.txt', 'corpus/valid_fr.txt', 'corpus/vocab.json')
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['valid_bsz'], shuffle=True)

    # Training loop
    encoder = Encoder(vocab.num_words, cfg['hidden_size'], device).to(device)
    decoder = Decoder(cfg['hidden_size'], vocab.num_words, device).to(device)
    enc_optimizer = optim.Adam(encoder.parameters())
    dec_optimizer = optim.Adam(decoder.parameters())
    criterion = nn.NLLLoss()

    for epoch in range(cfg['epochs']):
        encoder.train()
        decoder.train()
        loss_epoch = 0

        for i, (qc_idx, fr_idx, qc_len, fr_len) in enumerate(train_loader):
            if i > 10:
                break
            enc_hid = encoder.init_hidden(cfg['train_bsz']).to(device)
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            inp_qc = vocab.get_sentence(qc_idx)
            print('inp qc', ' '.join(inp_qc[0]))
            inp_fr = vocab.get_sentence(fr_idx)
            print('inp_fr', ' '.join(inp_fr[0]))

            qc_idx, fr_idx = qc_idx.to(device), fr_idx.to(device)

            inp_len = qc_idx.size(1)
            tgt_len = fr_idx.size(1)
            enc_outputs = torch.zeros(100, encoder.hidden_size, device=device)
            loss = 0

            for ei in range(inp_len):
                # print(qc_idx[:, ei].size())
                enc_out, enc_hid = encoder(qc_idx[:, ei], qc_len, enc_hid)
                print(enc_out)
                enc_outputs[ei] = enc_out[0, 0]

            dec_inp = torch.ones(cfg['train_bsz'], device=device) * 1
            dec_hid = enc_hid  # First decoder hidden state is last encoder hidden state
            for di in range(tgt_len):
                dec_out, dec_hid = decoder(dec_inp, dec_hid)

                # print(dec_out.type(), fr_idx[:, di].type())
                loss += criterion(dec_out, fr_idx[:, di].long())
                # if dec_out.item() == torch.tensor([[2]], device=device):  # ??? TODO: no hardcode for EOS
                #     break
                print(vocab.idx2word[str(torch.argmax(dec_out.detach()).item())])

            loss.backward(retain_graph=True)
            enc_optimizer.step()
            dec_optimizer.step()
            loss_batch = loss.item() / tgt_len
            loss_epoch += loss_batch
        print(loss_epoch)



    # initialization
    # encoder = Encoder(vocab.num_words, cfg['embedding_size'], cfg['hidden_size'], cfg['num_layers'])
    # decoder = Decoder(vocab.num_words, cfg['embedding_size'], cfg['hidden_size'], cfg['num_layers'])

    # seq2seq = Seq2Seq(encoder, decoder, vocab.num_words, device)
    # model = seq2seq.to(device)
    # optimizer = optim.Adam(model.parameters())
    # criterion = nn.CrossEntropyLoss()
    # for epoch in range(cfg['epochs']):
    #     model.train()
    #     total_loss = 0
    #     # TODO consider teacher forcing, gradient clipping,
    #     for idx, (qc_idx, fr_idx) in enumerate(train_loader):
    #         loss = 0
    #         # print(qc_idx)
    #         inp_qc = vocab.get_sentence(qc_idx)
    #         print(inp_qc)
    #         optimizer.zero_grad()
    #         qc_idx, fr_idx = qc_idx.to(device), fr_idx.to(device)
    #         print(qc_idx.type(), fr_idx.type())
    #         out = model(qc_idx, fr_idx)
    #         #out = model(qc_idx)
    #         # TODO reshape tensors
    #         out, fr_idx = out.float(), fr_idx.long()
    #         print(out.type(), fr_idx.type())

    #         print('-----')
    #         print(out, out.size())
    #         print(fr_idx, out.size())

    #         for i in range(out.size()[0]):
    #             print(i)
    #             print(out[idx, i], fr_idx[idx, i])
    #             loss += criterion(out[:,i], fr_idx[:,i])
    #         #loss = criterion(out, fr_idx)
    #         # break
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         break
        
    #     total_loss = total_loss/len(train_loader)
    #     print('total loss', total_loss)

