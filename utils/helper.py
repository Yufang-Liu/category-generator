import numpy as np
import re
import os

def read_turian(cfg):
    Turianword = []
    pwordlookup = []
    if os.path.isfile(cfg.PRE_EMB_FILE):
        with open(cfg.PRE_EMB_FILE, 'r') as fp:
            for w in fp:
                # w_list = re.split(r" +", w.strip())
                w_list = w.strip().split(' ')
                Turianword.append(w_list[0])
                pwordlookup.append([float(f) for f in w_list[1:] if f])
    return (Turianword, pwordlookup)

def top_k_2D_col_indexes(arr: np.array, k: int):
    assert (len(arr.shape) == 2 and k >= 0 and k <= arr.size)
    tot_size = arr.size
    num_row = arr.shape[0]
    t = np.argpartition(arr.T.reshape((tot_size,)), -k)[-k:]
    res = t // num_row
    index = t % num_row
    return res, index

class judge():
    def __init__(self, slash, slash_id, symbol, symbol_id, lparent_index, rparent_index, EOS_index, ccg_tag, ccg_tag_id):
        self.slash = slash
        self.slash_id = slash_id
        self.symbol = symbol
        self.symbol_id = symbol_id
        self.lparent_index = lparent_index
        self.rparent_index = rparent_index
        self.EOS_index = EOS_index
        self.parent = {'(', ')'}
        self.parent_id = {self.lparent_index, self.rparent_index}
        self.ccg_tag = ccg_tag
        self.ccg_tag_id = ccg_tag_id


    def __call__(self, lparent_cnt, rparent_cnt, last, ccg_id, check):
        if last == '#' and ccg_id in self.slash_id:
            return False
        if last in self.slash:
            if (ccg_id in self.slash_id) or (ccg_id in self.symbol_id):
                return False
            if ccg_id == self.rparent_index:
                return False
            if ccg_id == self.EOS_index:
                return False
        if last == '(' and ccg_id in self.slash_id:
            return False
        if last == ')':
            if ccg_id in self.ccg_tag_id or ccg_id == self.lparent_index:
                return False
        if last in self.ccg_tag:
            if ccg_id in self.ccg_tag_id or ccg_id == self.lparent_index:
                return False
        if last != '#' and ccg_id in self.symbol_id:
            return False
        if ccg_id == self.rparent_index:
            if lparent_cnt == rparent_cnt:
                return False
            if last == '(':
                return False
        if ccg_id == self.EOS_index:
            if lparent_cnt != rparent_cnt:
                return False
        if ccg_id in self.slash_id:
            temp = check + '1'
            if temp.find("0101") != -1:
                return False
        if ccg_id == self.rparent_index:
            lindex = check.rfind('(')
            temp = check[lindex+1:]
            if len(temp) == 1:
                return False
        return True

class FindCCG():
    def __init__(self, vocab):
        # slash indexes
        self.slash = {'/', '\\'}
        self.slash_list = set()
        for example in self.slash:
            self.slash_list.add(vocab.get_token_index(example, 'atom_ccg'))
        self.symbol = {',', '.', ':', ';'}
        self.symbol_list = set()
        # punctuation symbols
        for example in self.symbol:
            self.symbol_list.add(vocab.get_token_index(example, 'atom_ccg'))
        self.ccg_tag = set(vocab.vocab['atom_ccg'].keys()) - self.slash - self.symbol - {'EOS', '(', ')'}
        self.ccg_tag_id = set()
        for example in self.ccg_tag:
            self.ccg_tag_id.add(vocab.get_token_index(example, 'atom_ccg'))

        self.lparent_index = vocab.get_token_index('(', 'atom_ccg')
        self.rparent_index = vocab.get_token_index(')', 'atom_ccg')
        self.EOS_index = vocab.get_token_index('EOS', 'atom_ccg')

        self.judge_legal = judge(self.slash, self.slash_list, self.symbol, self.symbol_list, self.lparent_index, self.rparent_index, self.EOS_index, self.ccg_tag,
                            self.ccg_tag_id)

        self.tag_cnt = vocab.vocab_cnt['atom_ccg']
        self.vocab = vocab

    def initialize(self):
        # cnt used to add constraints on decoder
        self.last = '#'
        self.lparent_cnt = 0
        self.rparent_cnt = 0

        # string used for checking legal or not
        self.check = ''

    def __call__(self, y, ccg):
        while True:
            if not any(y):
                atom_ccg_id = 0
                while not self.judge_legal(self.lparent_cnt, self.rparent_cnt, self.last, atom_ccg_id, self.check):
                    atom_ccg_id += 1
                break
            atom_ccg_id = np.argmax(y)
            if atom_ccg_id == self.EOS_index and (not ccg):
                y[atom_ccg_id] = 0
                continue
            if self.last in self.symbol:
                atom_ccg_id = self.EOS_index
                break
            if not self.judge_legal(self.lparent_cnt, self.rparent_cnt, self.last, atom_ccg_id, self.check):
                y[atom_ccg_id] = 0
                continue
            break
        output = self.vocab.get_token_from_index(atom_ccg_id, 'atom_ccg')
        if atom_ccg_id == self.lparent_index:
            self.lparent_cnt += 1
        elif atom_ccg_id == self.rparent_index:
            self.rparent_cnt += 1
        self.last = output
        if atom_ccg_id in self.ccg_tag_id:
            self.check += '0'
        elif atom_ccg_id in self.slash_list:
            self.check += '1'
        elif output == '(':
            self.check += output
        elif output == ')':
            lindex = self.check.rfind('(')
            check = self.check[:lindex]
            self.check = check + '0'
        elif atom_ccg_id in self.symbol_list:
            self.check += output
        return atom_ccg_id

def get_ccg_matrix(vocab):
    res_list = []
    for i in range(vocab.vocab_cnt['ccg']):
        ccg = vocab.get_token_from_index(i, 'ccg')
        split_str = r'([()/\\])'
        atom_ccg_list = re.split(split_str, ccg)
        atom_ccg_list = [x for x in atom_ccg_list if x != '']
        atom_ccg_list.append('EOS')
        res = []
        for example in atom_ccg_list:
            res.append(vocab.get_token_index(example, 'atom_ccg'))
        res_list.append(res)
    return res_list
