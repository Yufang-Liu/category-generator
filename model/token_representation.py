# -- coding: utf-8 --
from typing import Dict, TypeVar, List
import numpy as np
import random
np.random.seed(666)
random.seed(666)
import _dynet as dy
from itertools import chain

Indices = TypeVar("Indices", List[int], List[List[int]])

class TokenRepresentation(object):
    '''
    n_word:int,单词的数目
    word_dim:int,单词的维度
    word:list,单词的列表
    pre_emb_file:预训练的词向量的文件
    '''

    def __init__(self, model, cfg, vocab, plist):

        pc = model.add_subcollection()
        word_dim = cfg.WORD_DIM
        n_word = vocab.vocab_cnt['my_word']
        n_pword = vocab.vocab_cnt['turian']
        char_dim = cfg.CHAR_DIM
        n_filter = cfg.N_FILTER
        win_sizes = [int(size) for size in str(cfg.WIN_SIZES).split('_')]
        n_char = vocab.vocab_cnt['my_char']
        if word_dim:
            self.wlookup = pc.lookup_parameters_from_numpy(
                np.random.randn(n_word, word_dim).astype(np.float32))
            pad_vec = [0.0 for _ in range(word_dim)]
            padding_index = vocab.get_token_index('*@PAD@*', 'my_word')
            self.wlookup.init_row(padding_index, pad_vec)

        if plist:
            pword_dim = len(plist[0])
            unk_pad_vec = [[0.0 for _ in range(pword_dim)]]
            pword_vec = unk_pad_vec + unk_pad_vec + plist
            parray = np.array(pword_vec, dtype=np.float32)/np.std(pword_vec)
            self.pwlookup = pc.lookup_parameters_from_numpy(parray.astype(np.float32))
            print('Load pre-trained word embedding. Vector dims %d, Word nums %d' % (pword_dim, n_pword))
        else:
            pword_dim = 0

        self.token_dim = word_dim + pword_dim

        self.Ws = [pc.add_parameters((char_dim, size, 1, n_filter),
                                     init=dy.GlorotInitializer(gain=0.5))
                   for size in win_sizes]
        self.win_sizes = win_sizes
        self.n_char = n_char
        self.char_dim = char_dim
        self.n_filter = n_filter
        self.clookup = pc.lookup_parameters_from_numpy(
            np.random.randn(n_char, char_dim).astype(np.float32))
        pad_vec = [0.0 for _ in range(char_dim)]
        padding_index = vocab.get_token_index('*@PAD@*', 'my_char')
        self.clookup.init_row(padding_index, pad_vec)
        self.token_dim += n_filter * len(win_sizes)

        if cfg.BERT_DIM:
            self.W = pc.add_parameters((cfg.BERT_DIM, 768), init='normal', mean=0, std=1)
            self.token_dim += cfg.BERT_DIM

        self.word_dim = word_dim
        self.pword_dim = pword_dim
        self.char_dim = char_dim
        self.pc = pc
        self.cfg = cfg
        self.spec = (cfg, vocab, plist)

    def __call__(
        self,
        indexes: Dict[str, List[Indices]],
        is_train = False) ->List[dy.Expression]:

        len_s = len(indexes['word']['my_word'][0])
        batch_num = len(indexes['word']['my_word'])

        vectors = []
        for i in range(len_s):
            # map token indexes to vector
            w_idxes = [indexes['word']['my_word'][x][i] for x in range(batch_num)]
            pw_idxes = [indexes['word']['turian'][x][i] for x in range(batch_num)]
            w_vec = dy.lookup_batch(self.wlookup, w_idxes)
            pw_vec = dy.lookup_batch(self.pwlookup, pw_idxes)

            # build token mask with dropout
            if is_train:
                wm = np.random.binomial(1, 1. - self.cfg.WORD_DROPOUT, batch_num).astype(np.float32)
                w_vec *= dy.inputTensor(wm, batched=True)
                pw_vec *= dy.inputTensor(wm, batched=True)

            c_idxes = [indexes['word']['my_char'][x][i] for x in range(batch_num)]
            maxn_char = len(c_idxes[0])
            c_idxes = list(chain.from_iterable(c_idxes))
            chars_emb = dy.lookup_batch(self.clookup, c_idxes)
            c2w = dy.reshape(chars_emb, (self.char_dim, maxn_char), batch_num)
            convds = [dy.conv2d(c2w, W, stride=(1, 1), is_valid=True) for W in self.Ws]
            poolds = [dy.maxpooling2d(convd, ksize=(1, maxn_char - win_size + 1), stride=(1, 1))
                      for win_size, convd in zip(self.win_sizes, convds)]
            actds = [dy.tanh(poold) for poold in poolds]
            words_batch = [dy.reshape(actd, (actd.dim()[0][2],)) for actd in actds]
            char_vec = dy.concatenate([out for out in words_batch])
            if is_train:
                cm = np.random.binomial(1, 1. - self.cfg.CHAR_DROPOUT, batch_num).astype(np.float32)
                scale = np.logical_or(np.logical_or(wm, wm), cm) * 4 / (2 * wm + wm + cm + 1e-12)
                cm *= scale
                char_vec *= dy.inputTensor(cm, batched=True)
            if self.cfg.BERT_DIM:
                bert_emb = np.array([indexes['bert'][x][i] for x in range(batch_num)]).transpose()
                bert_vec = self.W*dy.inputTensor(bert_emb, batched=True)
                if is_train:
                    wm = np.random.binomial(1, 1. - self.cfg.WORD_DROPOUT, batch_num).astype(np.float32)
                    bert_vec *= dy.inputTensor(wm, batched=True)
                vectors.append(dy.concatenate([w_vec, pw_vec, char_vec, bert_vec]))
            else:
                vectors.append(dy.concatenate([w_vec, pw_vec, char_vec]))
        return vectors


    @staticmethod
    def from_spec(spec, pc):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        cfg, vocab, plist = spec
        return TokenRepresentation(pc, cfg, vocab, plist)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc
