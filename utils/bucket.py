import numpy as np
import dynet as dy

class Bucket:
    def __init__(self, gener_decoder, mlp_decoder, vocab, model, beam_width=None, prob=None):
        self.gener_decoder = gener_decoder
        self.mlp_decoder = mlp_decoder
        self.vocab = vocab
        self.model = model
        self.beam_width = beam_width
        self.prob = prob
        self.gene_value = {}
        self.gene_output = {}

    def cal_loss(self, vectors, masks, indexes, is_train, accelerate=True,
                 dropout_x=None, dropout_h=None, ccg_dropout=None, mlp_dropout=None):
        truth_batch = np.transpose(np.array(indexes['ccg']['atom_ccg']))
        truth_batch_dim = truth_batch.shape
        sent_len = truth_batch_dim[1]
        truth_batch = np.reshape(truth_batch, (truth_batch_dim[0], truth_batch_dim[1] * truth_batch_dim[2]),
                                 order='F')
        arr = np.array(masks['2D'])
        masks_batch = np.transpose(arr)
        masks_batch_dim = masks_batch.shape
        masks_batch = np.reshape(masks_batch, (masks_batch_dim[0], masks_batch_dim[1] * masks_batch_dim[2]),
                                 order='F')
        full_mask = np.transpose(np.array(masks['1D']))
        full_mask_dim = full_mask.shape
        full_mask = np.reshape(full_mask, (full_mask_dim[0] * full_mask_dim[1],), order='F')
        full_truth = np.transpose(np.array(indexes['ccg']['ccg']))
        full_truth_dim = full_truth.shape
        full_truth = np.reshape(full_truth, (full_truth_dim[0] * full_truth_dim[1],), order='F')
        full_len = np.transpose(np.array(indexes['ccg']['length']))
        full_len_dim = full_len.shape
        full_len = np.reshape(full_len, (full_len_dim[0] * full_len_dim[1],), order='F')
        full_ccg = np.transpose(np.array(indexes['ccg']['full_ccg']))
        full_ccg_dim = full_ccg.shape
        full_ccg = np.reshape(full_ccg, (full_ccg_dim[0] * full_ccg_dim[1],), order='F')

        full_word = np.transpose(np.array(indexes['word']['my_word']))
        full_word_dim = full_word.shape
        full_word = np.reshape(full_word, (full_word_dim[0] * full_word_dim[1],), order='F')

        hidden_vectors = dy.concatenate_cols(vectors)
        if is_train:
            hidden_vectors = dy.dropout_dim(hidden_vectors, 1, mlp_dropout)
        h_dim = hidden_vectors.dim()
        h = dy.reshape(hidden_vectors, (h_dim[0][0], ), batch_size=h_dim[0][1]*h_dim[1])

        length_bucket = []
        length_list = [(idx, lens) for idx, lens in enumerate(full_len) if lens > 0]
        # remove unk
        if is_train and self.model == 'gener':
            unk_index = self.vocab.get_token_index('*@UNK@*', 'ccg')
            unk_list = []
            for k, v in length_list:
                if full_truth[k] == unk_index:
                    unk_list.append((k, v))
            for k, v in unk_list:
                length_list.remove((k, v))

        if is_train or accelerate:
            length_list_sort = sorted(length_list, key=lambda t: t[1])
            ccg_index = [idx for idx, lens in length_list_sort]
            bucket_size = 512 if self.model == 'mlp' else 256
            for beg in range(0, len(ccg_index), bucket_size):
                ins_batch = ccg_index[beg:beg + bucket_size]
                length_bucket.append((ins_batch, full_len[ins_batch[-1]]))
        else:
            ccg_index = [idx for idx, lens in length_list]
            bucket_size = 256 if self.model == 'mlp' else 256
            # if not accelerate, set bucket_size smaller
            for beg in range(0, len(ccg_index), bucket_size):
                ins_batch = ccg_index[beg:beg+bucket_size]
                length_bucket.append((ins_batch, masks_batch.shape[0]))

        if is_train:
            loss_bucket = []
            total_token = 0
        else:
            good = 0
            total = 0
        for index_list, lens in length_bucket:
            atom_truth = truth_batch[:, index_list]
            atom_mask = masks_batch[:, index_list]
            atom_truth = atom_truth[:lens]
            atom_mask = atom_mask[:lens]
            ccg_truth = full_truth[index_list]
            full_ccg_batch = full_ccg[index_list]
            ccg_mask = full_mask[index_list]
            h_bucket = dy.pick_batch_elems(h, index_list)
            word_bucket = full_word[index_list]

            #prepare sent vector
            # idx = np.array(index_list) if len(index_list) > 1 else np.array([index_list])
            # sent_vec = dy.pick_batch_elems(hidden_vectors, idx//sent_len)
            sent_vec = None

            word_index = np.array(index_list) % sent_len
            h_sent_info = (sent_vec, word_index, word_bucket)
            if is_train:
                if self.model == 'gener':
                    loss, token = self.gener_decoder(h_bucket, h_sent_info, self.vocab,
                                                     (atom_truth, atom_mask),
                                                     (ccg_truth, ccg_mask, full_ccg_batch),
                                                     True, accelerate, dropout_x,
                                                     dropout_h, ccg_dropout)
                    loss_bucket.append(loss)
                    total_token += token
                elif self.model == 'class':
                    loss, token = self.mlp_decoder(h_bucket, h_sent_info, (ccg_truth, ccg_mask), True)
                    loss_bucket.append(loss)
                    total_token += token
            else:
                if self.model == 'gener':
                    if self.beam_width == 1:
                        good_bucket, total_bucket = self.gener_decoder(h_bucket, h_sent_info,
                                                                       self.vocab,
                                                                       (atom_truth, atom_mask),
                                                                       (ccg_truth, ccg_mask, full_ccg_batch),
                                                                       False, accelerate)
                    else:
                        good_bucket, total_bucket = \
                            self.gener_decoder.beam_search(h_bucket, h_sent_info, self.vocab,
                                                           (atom_truth, atom_mask),
                                                           (ccg_truth, ccg_mask, full_ccg_batch),
                                                            self.beam_width)
                elif self.model == 'class':
                    if self.beam_width == 1:
                        good_bucket, total_bucket = self.mlp_decoder(h_bucket, h_sent_info,
                                                                     (ccg_truth, ccg_mask), False)
                    else:
                        good_bucket, total_bucket = self.mlp_decoder.beam_search(self.vocab,
                                                                                 h_bucket, h_sent_info,
                                                                                 (ccg_truth, ccg_mask),
                                                                                 self.beam_width)

                good += good_bucket
                total += total_bucket
        if is_train:
            return dy.esum(loss_bucket)/(total_token+1)*0.5
        else:
            if self.model == 'rerank':
                self.gene_output = {}
                self.gene_value = {}
            return good, total
