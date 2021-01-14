from overrides import overrides
from typing import Dict, List, Callable, TypeVar
from antu.io.vocabulary import Vocabulary
from antu.io.token_indexers.token_indexer import TokenIndexer
import re
Indeices = TypeVar("Indices", List[int], List[List[int]])

class AtomCCGIndexer(TokenIndexer):
    def __init__(
        self,
        related_vocabs: List[str],
        transform: Callable[[str, ], str] = lambda x:x,
        maxlen: int = 1)->None:
        self.related_vocabs = related_vocabs
        self.transform = transform
        self.maxlen = maxlen

    @overrides
    def count_vocab_items(
        self,
        token: str,
        counters: Dict[str, Dict[str, int]]) -> None:

        split_str = r'([()/\\])'

        for vocab_name in self.related_vocabs:
            if vocab_name in counters:
                atom_ccg_list = re.split(split_str, token)
                for atom_ccg in atom_ccg_list:
                    if atom_ccg:
                        counters[vocab_name][self.transform(atom_ccg)] += 1
                counters[vocab_name]['EOS'] += 1

    @overrides
    def tokens_to_indices(
        self,
        tokens: List[str],
        vocab: Vocabulary) -> Dict[str, List[List[int]]]:

        res = {}
        for vocab_name in self.related_vocabs:
            index_list = []
            split_str = r'([()/\\])'
            for token in tokens:
                atom_ccg_list = re.split(split_str, token)
                atom_ccg_list.append('EOS')
                index_list.append(
                    [vocab.get_token_index(self.transform(atom_ccg), vocab_name) for atom_ccg in atom_ccg_list if atom_ccg]
                )
            res[vocab_name] = index_list
        return res