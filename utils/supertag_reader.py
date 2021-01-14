#utf-8
import os, re, sys, pickle
from typing import List, Set
from overrides import overrides
from collections import Counter
from antu.io.fields.text_field import TextField
from antu.io.fields.meta_field import MetaField
from antu.io.vocabulary import Vocabulary
from antu.io.instance import Instance
from antu.io.dataset_readers.dataset_reader import DatasetReader
from antu.io.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from antu.io.token_indexers.char_token_indexer import CharTokenIndexer
from utils.DynamicTokenIndexer import DynamicTokenIndexer
from utils.atom_ccg_indexer import AtomCCGIndexer
from utils.Transform import Transform_for_Index, Transform_for_Count
from utils.FullCCGTokenIndexer import FullCCGTokenIndexer


def transform(word):
    if word == "-LRB-":
        word = "("
    elif word == "-RRB-":
        word = ")"
    elif word == "-LCB-":
        word = "{"
    elif word == "-RCB-":
        word = "}"
    elif word == "-LSB-":
        word = "["
    elif word == "-RSB-":
        word = "]"
    return word

def detransform(word):
    if word == "(":
        word = "-LRB-"
    elif word == ")":
        word = "-RRB-"
    elif word == "{":
        word = "-LCB-"
    elif word == "}":
        word = "-RCB-"
    elif word == "[":
        word = "-LSB-"
    elif word == "]":
        word = "-RSB-"
    return word

class SupertagReader(DatasetReader):
    def __init__(
        self,
        spacer,
        train: bool = False,
        pword: List[str] = None,
        vocab = None):

        self.spacer = spacer
        self.train = train
        self.Turianword = pword
        self.vocab = vocab

    def _isfloat(self, s):
        try:
            float(s)
            return True
        except:
            return False

    def _isfraction(self, s):
        try:
            index = s.find('/')
            firstnum = float(s[:index])
            lastnum = float(s[index + 2:])
            return True
        except:
            return False

    def _read(self, file_path: str) -> (List, List, List):
        files = os.listdir(file_path)
        for subfile in files:
            autofile = file_path + "/" + subfile
            autolist = os.listdir(autofile)
            for auto in autolist:
                filename_path = autofile + "/" + auto
                with open(filename_path, 'r') as fp:
                    tokens = None
                    for line in fp:
                        tok = re.split(self.spacer, line.strip())
                        if not tok or line.strip() == '':
                            if tokens:
                                yield tokens
                        else:
                            if (line[0] == 'I' and line[1] == 'D'):
                                if tokens:
                                    yield tokens
                                # tokens = [(tok[0][3:], '*word*', '*ccg*', '*pos*', '*ori_pos*', '*argcat*')]
                            else:
                                word_sent = []
                                ccg_sent = []
                                startindex = line.find('<')
                                endindex = -1
                                temp = []
                                while startindex != -1:
                                    endindex = line.find('>', endindex + 1)
                                    temp.append(line[startindex + 1:endindex])
                                    startindex = line.find('<', startindex + 1)
                                for example in temp:
                                    node = example.split(' ')
                                    if len(node) == 6:
                                        # tokens.append((wordcout, node[4], node[1], node[2], node[3], node[5]))
                                        word = node[4]
                                        if '-' or '/' in node[4]:
                                            word = re.sub('\d', '0', word)
                                        tempstr = word.replace(',', '')
                                        if self._isfloat(tempstr):
                                            word = '0'
                                        word_sent.append(word)
                                        ccg_sent.append(node[1])
                                tokens = (word_sent,  ccg_sent)
                    if len(tokens) > 1:
                        yield tokens

    def _read_from_conll(self, file_path: str):
        with open(file_path, 'r') as fp:
            for line in fp:
                orign_word = []
                word_sent = []
                ccg_sent = []
                token_list = line.strip().split(' ')
                for token in token_list:
                    tok = token.split('|')
                    word = tok[0]
                    if '-' or '/' in tok[0]:
                        word = re.sub('\d', '0', word)
                    tempstr = word.replace(',', '')
                    if self._isfloat(tempstr):
                        word = '0'
                    word_sent.append(word)
                    ccg_sent.append(tok[2])
                    orign_word.append(tok[0])
                tokens = (orign_word, word_sent, ccg_sent)
                if len(tokens) > 1:
                    yield tokens

    def read_bert_file(self, filename):
        dataset = []
        for file in filename:
            f = open(file, 'rb')
            data = pickle.load(f)
            f.close()
            dataset += data
        for i in range(len(dataset)):
            sent, emb = dataset[i]
            yield sent, emb

    def _readfrom_out_of_domain(self, file_path: str):
        with open(file_path, 'r') as fp:
            for line in fp:
                ori_sent = []
                word_sent = []
                ccg_sent = []
                token_list = line.strip().split(' ')
                for token in token_list:
                    tok = token.split('|')
                    word = tok[0]
                    if '-' or '/' in tok[0]:
                        word = re.sub('\d', '0', word)
                    tempstr = word.replace(',', '')
                    if self._isfloat(tempstr):
                        word = '0'
                    ori_sent.append(tok[0])
                    word_sent.append(word)
                    ccg_sent.append(tok[2])
                tokens = (ori_sent, word_sent, ccg_sent)
                if len(tokens) > 1:
                    yield tokens

    def read_from_OOD(self, file_path: str, dynamic_oracle: int = 1, bert_file=None)-> List[Instance]:
        res = []
        single_id_word = SingleIdTokenIndexer(['my_word', 'turian'])
        char_id = CharTokenIndexer(['my_char'])
        single_id_ccg = SingleIdTokenIndexer(['ccg'])
        full_ccg_id = FullCCGTokenIndexer(['full_ccg'])
        if bert_file:
            bert_info = self.read_bert_file(bert_file)
        if dynamic_oracle:
            transform_for_count = Transform_for_Count()
            transform_for_index = Transform_for_Index(self.vocab)
            atom_ccg = DynamicTokenIndexer(['atom_ccg'], transform_for_count, transform_for_index)
        else:
            atom_ccg = AtomCCGIndexer(['atom_ccg'])
        for ori_sent, sent, ccg in self._readfrom_out_of_domain(file_path):
            sentTextField = TextField('word', sent, [single_id_word, char_id])
            ccgTextField = TextField('ccg', ccg, [single_id_ccg, atom_ccg, full_ccg_id])
            if bert_file:
                token_list, token_emb = bert_info.__next__()
                if self.compare_sent(token_list, ori_sent):
                    emb_filed = MetaField('bert', token_emb)
                else:
                    print(token_list)
                    print(ori_sent)
                    exit(1)
            inst = Instance([sentTextField, ccgTextField]) if not bert_file else \
                Instance([sentTextField, ccgTextField, emb_filed])
            res.append(inst)
        return res

    def read_from_results(self, file_path):
        with open(file_path, 'r') as fp:
            sent = []
            label_sent = []
            cat_sent = [[] for _ in range(4)]
            score_sent = [[] for _ in range(4)]
            flag = 0
            for w in fp:
                temp = w[:-1].strip().split('\t')
                if len(temp) == 3:
                    if temp[1] == '0' and len(sent):
                        yield (sent, label_sent, cat_sent, score_sent)
                        sent = []
                        cat_sent = [[] for _ in range(4)]
                        score_sent = [[] for _ in range(4)]
                        label_sent = []
                    sent.append(temp[0])
                    label_sent.append(temp[2])
                    flag = 1
                    continue
                if flag == 1 and temp[0][0].isdigit() == False:
                    cat_sent[0].append(temp[0])
                    cat_sent[1].append(temp[1])
                    cat_sent[2].append(temp[2])
                    cat_sent[3].append(temp[3])
                    flag = 2
                    continue
                if flag == 2 and temp[-1][-1].isdigit():
                    score_sent[0].append(temp[0])
                    score_sent[1].append(temp[1])
                    score_sent[2].append(temp[2])
                    score_sent[3].append(temp[3])
                    flag = 0
                    continue
                yield (sent, label_sent, cat_sent, score_sent)

    def compare_sent(self, sent1, sent2):
        flag = 0
        for i in range(len(sent1)):
            if sent1[i] != sent2[i]:
                flag = 1
        return flag == 0

    @overrides
    def read(self, file_path: str, dynamic_oracle: int = 1, strategy="parentheses", num=0, bert_file=None):
        if self.train:
            counter = {"my_word": Counter(), 'my_char': Counter(),  "ccg": Counter(), "atom_ccg": Counter(),
                       "full_ccg": Counter()}
            vocab = Vocabulary()
            vocab.extend_from_pretrained_vocab({'turian': self.Turianword})
        res = []
        single_id_word = SingleIdTokenIndexer(['my_word', 'turian'])
        char_id = CharTokenIndexer(['my_char'])
        single_id_ccg = SingleIdTokenIndexer(['ccg'])
        full_ccg_id = FullCCGTokenIndexer(['full_ccg'])
        if dynamic_oracle:
            transform_for_count = Transform_for_Count(strategy=strategy, num=num)
            if self.train:
                transform_for_index = Transform_for_Index(vocab, strategy=strategy, num=num)
            else:
                transform_for_index = Transform_for_Index(self.vocab)
            atom_ccg = DynamicTokenIndexer(['atom_ccg'], transform_for_count, transform_for_index)
        else:
            atom_ccg = AtomCCGIndexer(['atom_ccg'])
        if bert_file:
            bert_info = self.read_bert_file(bert_file)
        for ori_sent, sent, ccg in self._read_from_conll(file_path):
            sentTextField = TextField('word', sent, [single_id_word, char_id])
            ccgTextField = TextField('ccg', ccg, [single_id_ccg, atom_ccg, full_ccg_id])
            if bert_file:
                token_list, token_emb = bert_info.__next__()
                if self.compare_sent(token_list, ori_sent):
                    emb_filed = MetaField('bert', token_emb)
                else:
                    print(token_list)
                    print(ori_sent)
                    exit(1)
            inst = Instance([sentTextField, ccgTextField]) if not bert_file else \
                Instance([sentTextField, ccgTextField, emb_filed])
            if self.train:
                inst.count_vocab_items(counter)
            res.append(inst)
        if self.train:
            min_count = {'my_word': 3, 'ccg':10}
            vocab.extend_from_counter(counter, min_count, no_unk_namespace={'full_ccg'})
            if dynamic_oracle:
                counter = transform_for_count.counter
                ccg_list = sorted(counter.items(), key=lambda d: d[1], reverse=True)
                for ccg, k in ccg_list:
                    if k >= 10:
                        vocab.add_token_to_namespace(ccg, 'atom_ccg')
            self.vocab = vocab
            return res, vocab
        else:
            return res

    @overrides
    def input_to_instance(self, inputs: (List[str], List[str], List[str])) -> Instance:
        pass