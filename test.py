# -- coding: utf-8 --
import random, sys, argparse, os
sys.path.append('/home/yfliu/software/antu-feature-dynamic_oracle_adaptation/')
random.seed(666)
import numpy as np
np.random.seed(666)
from utils.supertag_reader import SupertagReader
from antu.io.configurators.ini_configurator import IniConfigurator
from utils.dataset import Dataset
from utils.helper import *
import pickle
import time

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='configs/config.cfg')
    argparser.add_argument('--continue_training', action='store_true',
                           help='Load model and continue training')
    argparser.add_argument('--gpu', default='7', help='GPU ID(-1 to cpu)')
    argparser.add_argument('--model', default='generator',
                           help='run which model, generator or classifier')
    args, extra_args = argparser.parse_known_args()
    cfg = IniConfigurator(args.config_file, extra_args)
    # DyNet setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import dynet_config
    dynet_config.set(mem=cfg.DYNET_MEM, random_seed=cfg.DYNET_SEED)
    dynet_config.set_gpu()
    import dynet as dy
    from utils.bucket import Bucket

    # deal with the raw data
    Turianword, pwordlookup = read_turian(cfg)
    with open(cfg.VOCAB_FILE, 'rb') as fp:
        vocab = pickle.load(fp)
    spacer = r'[\s]'
    if cfg.BERT_DIM:
        file_name = '/home/yfliu/data_from_8014/bert_emb/processed/'
        test_bert_file = [file_name+'test.emb']
        add_bert = True
    else:
        test_bert_file = None
        add_bert = False
    supertag_reader_test = SupertagReader(spacer, pword=Turianword, vocab=vocab)
    testset = supertag_reader_test.read(cfg.TEST_FILE, cfg.DYNAMIC_ORACLE,
                                        bert_file=test_bert_file)
    test_datasets = Dataset(vocab, testset, cfg.DYNAMIC_ORACLE, add_bert=add_bert)

    pc = dy.ParameterCollection()
    token_repre, encoder, decoder = dy.load(cfg.MODEL_FILE, pc)
    def cmp(ins): return len(ins['word'].tokens)

    time_start = time.time()
    test_batch = test_datasets.get_batches(cfg.TEST_BATCH_SIZE, True, cmp, False)
    bucket = Bucket(decoder, None, vocab, "gener", beam_width=cfg.BEAM_WIDTH) \
             if args.model == 'generator' else \
             Bucket(None, decoder, vocab, "class", beam_width=cfg.BEAM_WIDTH)

    good = 0
    total = 0
    for indexes, masks in test_batch:
        dy.renew_cg()
        vectors = token_repre(indexes, False)
        vectors = encoder(vectors, None, cfg.RNN_X_DROPOUT, cfg.RNN_H_DROPOUT, False)
        # when need to output the predict or record test speed, accelerate should be set to False
        good_batch, total_batch = bucket.cal_loss(vectors, masks, indexes, False,
                                                  accelerate=True)
        good += good_batch
        total += total_batch
    acc = good * 1.0 / total
    print(acc)
    time_end = time.time()
    print('totally cost', time_end - time_start)

if __name__ == '__main__':
    main()
