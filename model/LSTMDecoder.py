import random
random.seed(666)
import dynet as dy
import numpy as np
np.random.seed(666)
import heapq
from utils.helper import *

class LSTMDecoder(object):
    def __init__(self, model, x_dims, h_dims, ccg_dims, LSTMBuilder, n_tag):
        pc = model.add_subcollection()
        input_dim = x_dims + ccg_dims
        #decoder lstm
        self.f = LSTMBuilder(1, input_dim, h_dims, pc)
        self.W = pc.add_parameters((n_tag, h_dims), init='normal', mean=0, std=1)
        # lookup table
        self.ccg_lookup = pc.lookup_parameters_from_numpy(
            np.random.randn(n_tag, ccg_dims).astype(np.float32))

        self.pc = pc
        self.spec = (x_dims, h_dims, ccg_dims, LSTMBuilder, n_tag)
        self.h_dim = h_dims
        self.ccg_dims = ccg_dims
        self.ntags = n_tag

    def __call__(self, hidden_vectors, h_sent_info, vocab, atom_info, ccg_info, train,
                 accelerate=True, dropout_x=None, dropout_h=None, ccg_dropout=None):
        _, word_id, word_index = h_sent_info
        truth_batch, masks_batch = atom_info
        full_truth, full_mask, full_ccg = ccg_info
        maxlen = truth_batch.shape[0]
        batch_size = len(full_truth)
        mask_dim = masks_batch.shape
        arr = np.reshape(masks_batch, (mask_dim[0]*mask_dim[1], ))
        total_token = np.sum(arr)*1.0

        unk_index = vocab.get_token_index('*@UNK@*', 'ccg')
        if train:
            loss = []
        else:
            output = ['' for _ in range(batch_size)]
            flag = 1 - full_mask
            predict = None
            if not accelerate:
                maxlen = 125  # the maximum sequence length
        f = None
        atom_cnt = 0
        start_flag = 0
        while atom_cnt < maxlen:
            if start_flag == 0:
                f = self.f.initial_state()
                if train:
                    self.f.set_dropouts(dropout_x, dropout_h)
                    self.f.set_dropout_masks(batch_size)
                else:
                    self.f.set_dropouts(0.0, 0.0)
                    self.f.set_dropout_masks(batch_size)
                ccg_vec_batch = [0] * batch_size
                ccg_vec = dy.lookup_batch(self.ccg_lookup, ccg_vec_batch)
            else:
                if train:
                    ccg_vec = dy.lookup_batch(self.ccg_lookup, truth_batch[atom_cnt - 1])
                    cm = np.random.binomial(1, 1. - ccg_dropout, batch_size).astype(np.float32)
                    ccg_vec *= dy.inputTensor(cm, batched=True)
                else:
                    if batch_size > 1:
                        ccg_vec = dy.lookup_batch(self.ccg_lookup, predict)
                    else:
                        ccg_vec = dy.lookup_batch(self.ccg_lookup, [predict])
            x = dy.concatenate([hidden_vectors, ccg_vec])
            f = f.add_input(x)
            o = f.output()
            y = self.W * o
            if train:
                errors = dy.pickneglogsoftmax_batch(y, truth_batch[atom_cnt])
                m = np.reshape(masks_batch[atom_cnt], (1, batch_size), order='F')
                m = dy.inputTensor(m, True)
                err = dy.sum_batches(dy.cmult(errors, m))
                loss.append(err)
            else:
                predict = np.argmax(y.npvalue(), axis=0)
                for i in range(batch_size):
                    if batch_size > 1:
                        temp = vocab.get_token_from_index(predict[i], 'atom_ccg')
                    else:
                        temp = vocab.get_token_from_index(predict, 'atom_ccg')
                    if temp == 'EOS' and flag[i] == 0:
                        flag[i] = 1
                    elif temp != 'EOS' and flag[i] == 0:
                        output[i] += temp
                        if not accelerate and not vocab.get_token_from_index(full_ccg[i], 'full_ccg').startswith(output[i]):
                            flag[i] = 1
                if (flag == 1).all():
                    break
            atom_cnt += 1
            start_flag = 1
        if train:
            return dy.esum(loss), total_token
        else:
            good = 0
            for i in range(batch_size):
                if output[i] == vocab.get_token_from_index(full_ccg[i], 'full_ccg') and full_mask[i] == 1:
                    good += 1
            return good, np.sum(full_mask)

    def softmax(self, x):
        """Compute the softmax in a numerically stable way."""
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def beam_search(self, hidden_vectors, h_sent_info, vocab, atom_info, ccg_info, beam_width):
        _, word_id, word_index = h_sent_info
        truth_batch, masks_batch = atom_info
        full_truth, full_mask, full_ccg = ccg_info
        maxlen = truth_batch.shape[0]
        eos_index = vocab.get_token_index('EOS', 'atom_ccg')
        batch_size = truth_batch.shape[1]
        output = [['' for _ in range(batch_size)] for _ in range(beam_width)]
        flag = [[0 for _ in range(batch_size)] for _ in range(beam_width)]
        flag = np.array(flag)
        ccg_len = [[0 for _ in range(batch_size)] for _ in range(beam_width)]
        best_value = None
        best_index = None
        maxlen = 125
        f = None
        atom_cnt = 0
        start_flag = 0
        while atom_cnt < maxlen:
            if start_flag == 0:
                f = self.f.initial_state()
                self.f.set_dropouts(0.0, 0.0)
                self.f.set_dropout_masks(batch_size)
                ccg_vec_batch = [0] * batch_size
                ccg_vec = dy.lookup_batch(self.ccg_lookup, ccg_vec_batch)
            else:
                ccg_vec = []
                for i in range(beam_width):
                    ccg_vec.append(dy.lookup_batch(self.ccg_lookup, best_index[:, i]))
                ccg_vec = dy.concatenate_cols(ccg_vec)
            x = dy.concatenate([hidden_vectors, ccg_vec])
            f = f.add_input(x)
            o = f.output()
            if start_flag == 0:
                y = dy.log(dy.softmax(self.W * o))
                y_value = y.npvalue()
                best_index = []
                if batch_size > 1:
                    for i in range(batch_size):
                        y_list = list(y_value[:, i])
                        y_list[eos_index] = -1e9
                        best_index.append(list(map(y_list.index, heapq.nlargest(beam_width, y_list))))
                else:
                    y_list = list(y_value)
                    y_list[eos_index] = -1e9
                    best_index.append(list(map(y_list.index, heapq.nlargest(beam_width, y_list))))
                best_index = np.array(best_index)
                best_value = []
                for i in range(beam_width):
                    best_value.append(dy.pick_batch(y, best_index[:, i]).npvalue())
                best_value = np.array(best_value)
                best_value = np.reshape(best_value, (beam_width, batch_size), order='F')
                for i in range(batch_size):
                    for k in range(beam_width):
                        temp = vocab.get_token_from_index(best_index[i][k], 'atom_ccg')
                        if temp == 'EOS' and flag[k][i] == 0:
                            flag[k][i] = 1
                            ccg_len[k][i] += 1
                        elif temp != 'EOS' and flag[k][i] == 0:
                            output[k][i] += temp
                            ccg_len[k][i] += 1
                flag = np.array(flag)
                output = np.array(output)
                ccg_len = np.array(ccg_len)
                hidden_vectors = dy.concatenate_cols([hidden_vectors for _ in range(beam_width)])
                h0, c0 = f.s()
                h1 = [h0 for _ in range(beam_width)]
                h1 = dy.concatenate_cols(h1)
                c1 = [c0 for _ in range(beam_width)]
                c1 = dy.concatenate_cols(c1)
                f = f.set_s((h1, c1))
            else:
                y = dy.log(dy.softmax(self.W * o))
                y_value = y.npvalue()
                if batch_size > 1:
                    for i in range(batch_size):
                        for k in range(beam_width):
                            if flag[k][i] == 0:
                                y_value[:, k, i] += best_value[k][i] * pow(ccg_len[k][i], 0.15)
                                y_value[:, k, i] = y_value[:, k, i] / (pow(ccg_len[k][i] + 1, 0.15))
                            else:
                                y_value[:, k, i] = -np.e ** 100
                                y_value[0, k, i] = best_value[k][i]
                else:
                    for k in range(beam_width):
                        if flag[k][0] == 0:
                            y_value[:, k] += best_value[k][0] * pow(ccg_len[k][0], 0.15)
                            y_value[:, k] = y_value[:, k] / (pow(ccg_len[k][0] + 1, 0.15))
                        else:
                            y_value[:, k] = -np.e ** 100
                            y_value[0, k] = best_value[k][0]
                best_value = []
                best_beam = []
                best_index = []
                for i in range(batch_size):
                    if batch_size > 1:
                        b = y_value[:, :, i]
                    else:
                        b = y_value
                    b_b, b_i = top_k_2D_col_indexes(b, beam_width)
                    best_beam.append(b_b)
                    best_index.append(b_i)
                    b_v = []
                    for j in range(beam_width):
                        b_v.append(b[b_i[j]][b_b[j]])
                    best_value.append(b_v)
                best_index = np.array(best_index)
                best_beam = np.array(best_beam)
                best_value = np.transpose(np.array(best_value))
                output_new = [['' for _ in range(batch_size)] for _ in range(beam_width)]
                flag_new = [[0 for _ in range(batch_size)] for _ in range(beam_width)]
                ccg_len_new = [[0 for _ in range(batch_size)] for _ in range(beam_width)]
                for i in range(batch_size):
                    for k in range(beam_width):
                        temp = vocab.get_token_from_index(best_index[i][k], 'atom_ccg')
                        ori_beam = best_beam[i][k]
                        if flag[ori_beam][i] == 1:
                            flag_new[k][i] = 1
                            output_new[k][i] = output[ori_beam][i]
                            ccg_len_new[k][i] = ccg_len[ori_beam][i]
                        elif temp == 'EOS' and flag[ori_beam][i] == 0:
                            flag_new[k][i] = 1
                            output_new[k][i] = output[ori_beam][i]
                            ccg_len_new[k][i] = ccg_len[ori_beam][i] + 1
                        elif temp != 'EOS' and flag[ori_beam][i] == 0:
                            flag_new[k][i] = 0
                            output_new[k][i] = output[ori_beam][i] + temp
                            ccg_len_new[k][i] = ccg_len[ori_beam][i] + 1
                flag = np.array(flag_new)
                output = np.array(output_new)
                ccg_len = np.array(ccg_len_new)
                h0, c0 = f.s()
                h1 = []
                c1 = []
                for i in range(beam_width):
                    h1.append(dy.pick_batch(h0, best_beam[:, i], dim=1))
                    c1.append(dy.pick_batch(c0, best_beam[:, i], dim=1))
                h1 = dy.concatenate_cols(h1)
                c1 = dy.concatenate_cols(c1)
                f = f.set_s((h1, c1))
            if (flag == 1).all():
                break
            start_flag = 1
            atom_cnt += 1
        good = 0
        for i in range(batch_size):
            best_id = np.argmax(best_value[:, i])
            if output[best_id][i] == vocab.get_token_from_index(full_ccg[i], 'full_ccg') and full_mask[i] == 1:
                good += 1
            if full_mask[i] == 1:
                res = []
                value = np.exp(best_value[:, i] * np.power(ccg_len[:, i], 0.15))
                out_set = set()
                for bw in range(beam_width):
                    if output[bw][i] not in out_set:
                        out_set.add(output[bw][i])
                        res.append([output[bw][i], value[bw]])
        return good, np.sum(full_mask)

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instance with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        x_dims, h_dims, ccg_dims, LSTMBuilder, n_tag = spec
        return LSTMDecoder(model, x_dims, h_dims, ccg_dims, LSTMBuilder, n_tag)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc

