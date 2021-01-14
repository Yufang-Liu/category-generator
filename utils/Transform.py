import random, re
from typing import List

class Transform_for_Count():
    def __init__(self, strategy="parentheses", num=0):
        self.counter = {}
        self.strategy = strategy
        self.num = num

    def __call__(self, ccg_str) -> List[str]:
        split_str = r'([()/\\])'
        atom_ccg_list = re.split(split_str, ccg_str)
        atom_ccg_list = [x for x in atom_ccg_list if x != '']
        if self.strategy == "parentheses":
            atom_ccg_list.append('EOS')
            length = len(atom_ccg_list)
            parent_index = []
            parent_num = 0
            s = []
            if '(' in atom_ccg_list:
                for i in range(length):
                    if atom_ccg_list[i] == '(':
                        s.append(i)
                        parent_num = 0
                    elif atom_ccg_list[i] == ')':
                        parent_num += 1
                        parent_index.append((s[-1], i, parent_num))
                        s.pop()
                for begin, end, num in parent_index:
                    temp_str = "".join(atom_ccg_list[begin+1:end])
                    if temp_str in self.counter.keys():
                        self.counter[temp_str] += 1
                    else:
                        self.counter[temp_str] = 1
            return atom_ccg_list
        elif self.strategy == "ngram" and self.num:
            for i in range(len(atom_ccg_list)//self.num):
                temp_str = "".join(atom_ccg_list[self.num*i:self.num*(i+1)])
                if temp_str in self.counter.keys():
                    self.counter[temp_str] += 1
                else:
                    self.counter[temp_str] = 1
            atom_ccg_list.append('EOS')
            return atom_ccg_list


class Transform_for_Index():
    def __init__(self, vocab, strategy="parentheses", num=0):
        self.vocab = vocab
        self.strategy = strategy
        self.num = num

    def __call__(self, ccg_str) -> List[str]:
        split_str = r'([()/\\])'
        unk_index = self.vocab.get_token_index('*@UNK@*', 'atom_ccg')
        atom_ccg_list = re.split(split_str, ccg_str)
        atom_ccg_list = [x for x in atom_ccg_list if x != '']
        not_used = []
        if self.strategy == "parentheses":
            atom_ccg_list.append('EOS')
            length = len(atom_ccg_list)
            parent_index = []
            parent_num = 0
            s = []
            if '(' not in atom_ccg_list:
                res = atom_ccg_list
            else:
                for i in range(length):
                    if atom_ccg_list[i] == '(':
                        s.append(i)
                        parent_num = 0
                    elif atom_ccg_list[i] == ')':
                        parent_num += 1
                        parent_index.append((s[-1], i, parent_num))
                        s.pop()
                index = 0
                parent_index = sorted(parent_index)
                for begin, end, num in parent_index:
                    if begin < index:
                        continue
                    temp_str = "".join(atom_ccg_list[begin+1:end])
                    if self.vocab.get_token_index(temp_str, 'atom_ccg') != unk_index:
                        seed = random.uniform(0, 1)
                        if seed > 0.2:
                            atom_ccg_list[begin+1] = temp_str
                            for j in range(begin+2, end):
                                not_used.append(j)
                            index = end + 1
                res = []
                for i in range(length):
                    if i not in not_used:
                        res.append(atom_ccg_list[i])
            return res
        elif self.strategy == "ngram" and self.num:
            for i in range(len(atom_ccg_list) // self.num):
                temp_str = "".join(atom_ccg_list[self.num * i: self.num * (i+1)])
                if self.vocab.get_token_index(temp_str, 'atom_ccg') != unk_index:
                    seed = random.uniform(0, 1)
                    if seed > 0.0:
                        atom_ccg_list[self.num*i] = temp_str
                        for j in range(self.num*i+1, self.num*(i+1)):
                            not_used.append(j)
            atom_ccg_list.append('EOS')
            res = []
            for i in range(len(atom_ccg_list)):
                if i not in not_used:
                    res.append(atom_ccg_list[i])
            return res


#transform_for_count = Transform_for_Count(4)
#transform_for_count(r'(((NP/S)/((NP\S)/NP))/NP)\NP')
#
#transform_for_index = Transform_for_Index(4)
#for j in range(100):
#    transform_for_index(r'(((NP/S)/((NP\S)/NP))/NP)\NP')'''