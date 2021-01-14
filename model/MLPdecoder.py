import numpy as np
import _dynet as dy
import heapq

class MLP(object):

    def __init__(self, model, ntags, hidden_size):
        pc = model.add_subcollection()
        self.W = pc.add_parameters((ntags, hidden_size), init='normal', mean=0, std=1)
        self.pc = pc
        self.spec = (ntags, hidden_size)
        self.ntags = ntags

    def __call__(self, hidden_vectors, h_sent_info, ccg_info,  is_train):
        truth_batch, full_mask = ccg_info
        batch_size = len(full_mask)
        total_token = np.sum(full_mask)*1.0
        out = self.W * hidden_vectors
        if is_train:
            errors = dy.pickneglogsoftmax_batch(out, truth_batch)
            m = dy.inputTensor(full_mask, True)
            loss = dy.sum_batches(dy.cmult(errors, m))
            return loss, total_token
        else:
            good = 0
            pred = np.argmax(out.npvalue(), axis=0).transpose()
            for i in range(batch_size):
                if pred[i] == truth_batch[i] and full_mask[i] == 1:
                    good += 1
            return good, total_token

    def beam_search(self, vocab, hidden_vectors, h_sent_info, ccg_info, beam_width=None):
        _, word_id, word_index = h_sent_info
        truth_batch, full_mask = ccg_info
        batch_size = len(full_mask)
        total_token = np.sum(full_mask) * 1.0
        out = dy.softmax(self.W * hidden_vectors)
        good = 0
        pred = np.argmax(out.npvalue(), axis=0).transpose()
        for i in range(batch_size):
            value = out.npvalue()[:, i]
            index_list = list(map(list(value).index, heapq.nlargest(beam_width, list(value))))
            res = []
            for bw in range(beam_width):
                res.append([vocab.get_token_from_index(index_list[bw], 'ccg'), value[index_list[bw]]])
            if pred[i] == truth_batch[i] and full_mask[i] == 1:
                good += 1
        return good, total_token

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.
        It is one of the prerequisites for Dynet save/load method.
        """
        ntags, hidden_size = spec
        return MLP(model, ntags, hidden_size)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.
        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc
