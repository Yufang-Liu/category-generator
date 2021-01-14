import sys
import random
from typing import List, Callable
from antu.io.vocabulary import Vocabulary
from antu.io.instance import Instance

def shahow_padding(batch_input, vocab, add_bert = False):
    maxlen = 0
    maxlen_word = 3
    maxlen_ccg = 0
    for ins in batch_input:
        maxlen = max(maxlen, len(ins['word'].indexes['my_word']))
        for i in range(len(ins['word'].indexes['my_word'])):
            maxlen_word = max(maxlen_word, len(str(ins['word'].tokens[i])))
            maxlen_ccg = max(maxlen_ccg, len(ins['ccg'].indexes['atom_ccg'][i]))

    masks = {'1D': list(), '2D': list(), 'flat': list()}
    inputs = {
        'word': {'my_word': [], 'turian': [], 'my_char': []},
        'ccg': {'ccg': [], 'atom_ccg': [], 'length': [], 'full_ccg': []}
    }
    if add_bert:
        inputs['bert'] = []
    for ins in batch_input:
        word_len = len(ins['word'].indexes['my_word'])
        padding_length = maxlen - word_len
        #PAD word
        padding_index = vocab.get_token_index('*@PAD@*', 'my_word')
        pad_seq = [padding_index] * padding_length
        inputs['word']['my_word'].append(ins['word'].indexes['my_word'] + pad_seq)
        if add_bert:
            #PAD bert_emb
            pad_seq = [[0 for _ in range(768)]] * padding_length
            bert_sent = ins['bert'].indexes + pad_seq
            inputs['bert'].append(bert_sent)
        #PAD pertrained word
        padding_index = vocab.get_token_index('*@PAD@*', 'turian')
        pad_seq = [padding_index] * padding_length
        inputs['word']['turian'].append(ins['word'].indexes['turian'] + pad_seq)
        #pad char and atom_ccg
        padding_char_index = vocab.get_token_index('*@PAD@*', 'my_char')
        padding_atom_index = vocab.get_token_index('*@PAD@*', 'atom_ccg')
        char_vec = []
        atom_ccg_vec = []
        atom_ccg_masks = []
        ccg_len = []
        for i in range(maxlen):
            if i < word_len:
                #deal with char
                pad_char_length = maxlen_word - len(str(ins['word'].tokens[i]))
                pad_seq = [padding_char_index] * pad_char_length
                temp_char = ins['word'].indexes['my_char'][i] + pad_seq
                #deal with atom_ccg
                padding_atom_length = maxlen_ccg - len(ins['ccg'].indexes['atom_ccg'][i])
                pad_seq = [padding_atom_index] * padding_atom_length
                temp_atom = ins['ccg'].indexes['atom_ccg'][i] + pad_seq
                atom_mask = [1]*(maxlen_ccg - padding_atom_length) + [0]*padding_atom_length
                #add length
                ccg_len.append(len(ins['ccg'].indexes['atom_ccg'][i]))
            else:
                #deal with char
                pad_seq = [padding_char_index] * maxlen_word
                temp_char = pad_seq
                #deal with atom
                pad_seq = [padding_atom_index] * maxlen_ccg
                temp_atom = pad_seq
                atom_mask = [0] * maxlen_ccg
                #add_length
                ccg_len.append(0)
            char_vec.append(temp_char)
            atom_ccg_vec.append(temp_atom)
            atom_ccg_masks.append(atom_mask)
        inputs['word']['my_char'].append(char_vec)
        inputs['ccg']['atom_ccg'].append(atom_ccg_vec)
        inputs['ccg']['length'].append(ccg_len)
        masks['2D'].append(atom_ccg_masks)
        #PAD ccg
        padding_index = vocab.get_token_index('*@PAD@*', 'ccg')
        pad_seq = [padding_index] * padding_length
        inputs['ccg']['ccg'].append(ins['ccg'].indexes['ccg']+pad_seq)
        #PAD full ccg
        padding_index = vocab.get_token_index('*@PAD@*', 'full_ccg')
        pad_seq = [padding_index] * padding_length
        inputs['ccg']['full_ccg'].append(ins['ccg'].indexes['full_ccg']+pad_seq)
        #mask
        ins_mask = [1]*(maxlen-padding_length) + [0] *padding_length
        masks['1D'].append(ins_mask)

    #build [Flat] masks
    for ins in masks['1D']:
        masks['flat'].extend(ins)
    return inputs, masks

class Dataset:
    def __init__(
        self,
        vocab : Vocabulary,
        dataset : List,
        dynamic_oracle: int = 1,
        add_bert = False):
        self.vocab = vocab
        self.datasets = dataset
        self.is_padding = False
        self.dynamic_oracle = dynamic_oracle
        self.add_bert = add_bert
        self.pad_dataset = {}

    def get_batches(
        self,
        size: int,
        ordered: bool = False,
        cmp: Callable[[Instance, Instance], int] = None,
        is_infinite: bool=False) -> List[List[int]]:
        if ordered:
            self.datasets.sort(key=cmp)

        num = len(self.datasets) #Number of Instance
        if is_infinite and self.dynamic_oracle:
            while is_infinite:
                random.shuffle(self.datasets)
                for beg in range(0, num, size):
                    ins_batch = self.datasets[beg:beg+size]
                    for ins in ins_batch: ins.dynamic_index_fields(self.vocab, ['atom_ccg'])
                    indexes, masks = shahow_padding(ins_batch, self.vocab, self.add_bert)
                    yield indexes, masks
        elif is_infinite:
                result = []
                for beg in range(0, num, size):
                    ins_batch = self.datasets[beg:beg + size]
                    for ins in ins_batch: ins.index_fields(self.vocab)
                    indexes, masks = shahow_padding(ins_batch, self.vocab, self.add_bert)
                    yield indexes, masks
                    result.append((indexes, masks))
        else:
            result = []
            if not self.is_padding:
                for beg in range(0, num, size):
                    ins_batch = self.datasets[beg:beg+size]
                    for ins in ins_batch: ins.index_fields(self.vocab)
                    indexes, masks = shahow_padding(ins_batch, self.vocab, self.add_bert)
                    result.append((indexes, masks))
                self.is_padding = True
                self.pad_dataset = result
            for indexes, masks in self.pad_dataset:
                yield indexes, masks

        if self.dynamic_oracle == 0:
            while is_infinite:
                random.shuffle(result)
                for indexes, masks in result:
                    yield indexes, masks
