# -- coding: utf-8 --
import random, sys, argparse, os

sys.path.append('/home/yfliu/software/antu-feature-dynamic_oracle_adaptation/')
random.seed(666)
import numpy as np

np.random.seed(666)
import pickle
from utils.supertag_reader import SupertagReader
from antu.io.configurators.ini_configurator import IniConfigurator
from antu.utils.dual_channel_logger import dual_channel_logger
from utils.dataset import Dataset
from utils.helper import *


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='configs/config.cfg')
    argparser.add_argument('--continue_training', action='store_true',
                           help='Load model and continue training')
    argparser.add_argument('--name', default='experiment',
                           help='The name of the experiment.')
    argparser.add_argument('--gpu', default='0', help='GPU ID(-1 to cpu)')
    argparser.add_argument('--model', default='generator',
                           help='run which model, generator or classifier')
    args, extra_args = argparser.parse_known_args()
    cfg = IniConfigurator(args.config_file, extra_args)

    # Logger setting
    logger = dual_channel_logger(
        __name__,
        file_path=cfg.LOG_FILE,
        file_model='w',
        formatter='%(asctime)s - %(levelname)s - %(message)s',
        time_formatter='%m-%d %H:%M')

    # DyNet setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import dynet_config
    dynet_config.set(mem=cfg.DYNET_MEM, random_seed=cfg.DYNET_SEED)
    dynet_config.set_gpu()
    import dynet as dy
    from model.token_representation import TokenRepresentation
    from antu.nn.dynet.seq2seq_encoders.rnn_builder import DeepBiLSTMBuilder, \
        orthonormal_VanillaLSTMBuilder
    from model.LSTMDecoder import LSTMDecoder
    from model.MLPdecoder import MLP
    from utils.bucket import Bucket

    # deal with the raw data
    Turianword, pwordlookup = read_turian(cfg)
    spacer = r'[\s]'
    if cfg.BERT_DIM:
        file_name = '/home/yfliu/data_from_8014/bert_emb/processed/'
        train_bert_file = [file_name + 'train0.emb', file_name + 'train1.emb',
                           file_name + 'train2.emb', file_name + 'train3.emb']
        dev_bert_file = [file_name + 'dev.emb']
        add_bert = True
    else:
        train_bert_file = None
        dev_bert_file = None
        add_bert = False
    supertag_reader = SupertagReader(spacer, train=True, pword=Turianword)
    trainset, vocab = supertag_reader.read(cfg.TRAIN_FILE, cfg.DYNAMIC_ORACLE,
                                           strategy=cfg.STRATEGY, num=cfg.NUM,
                                           bert_file=train_bert_file)
    supertag_reader_dev = SupertagReader(spacer, pword=Turianword, vocab=vocab)
    devset = supertag_reader_dev.read(cfg.DEV_FILE, cfg.DYNAMIC_ORACLE,
                                      strategy=cfg.STRATEGY, num=cfg.NUM,
                                      bert_file=dev_bert_file)
    random.shuffle(trainset)
    train_datasets = Dataset(vocab, trainset, cfg.DYNAMIC_ORACLE, add_bert)
    dev_datasets = Dataset(vocab, devset, cfg.DYNAMIC_ORACLE, add_bert)

    def cmp(ins):
        return len(ins['word'].tokens)

    train_batch = train_datasets.get_batches(cfg.TRAIN_BATCH_SIZE, True, cmp, True)
    print("train: " + str(len(trainset)))
    print("dev: " + str(len(devset)))

    pc = dy.ParameterCollection()
    trainer = dy.AdamTrainer(
        pc,
        alpha=cfg.LEARNING_RATE,
        beta_1=cfg.ADAM_BETA1,
        beta_2=cfg.ADAM_BETA2,
        eps=cfg.EPS)
    token_repre = TokenRepresentation(pc, cfg, vocab, pwordlookup)
    encoder = DeepBiLSTMBuilder(
        pc,
        cfg.ENC_LAYERS,
        token_repre.token_dim,
        cfg.ENC_DIMS,
        orthonormal_VanillaLSTMBuilder,
        param_init=True,
        fb_fusion=True)
    decoder = LSTMDecoder(
        pc,
        cfg.ENC_DIMS * 2,
        cfg.DEC_DIMS,
        cfg.DEC_CCG_DIMS,
        orthonormal_VanillaLSTMBuilder,
        vocab.vocab_cnt['atom_ccg']) \
        if args.model == 'generator' \
        else MLP(
        pc,
        vocab.vocab_cnt['ccg'],
        cfg.ENC_DIMS * 2)
    bucket = Bucket(decoder, None, vocab, "gener", beam_width=cfg.BEAM_WIDTH) \
              if args.model == 'generator' else \
            Bucket(None, decoder, vocab, "class", beam_width=cfg.BEAM_WIDTH)

    # train model
    cnt_iter = 0
    valid_loss = []
    BEST_DEV_ACC = 0
    logger.info("Experiment name: %s" % args.name)
    logger.info('Git SHA: %s' % os.popen('git log -1 | head -n 1 | cut -c 8-13').
                readline().rstrip())
    while cnt_iter < cfg.MAX_ITER:
        dy.renew_cg()
        cnt_iter += 1
        indexes, masks = train_batch.__next__()
        vectors = token_repre(indexes, True)
        vectors = encoder(vectors, None, cfg.RNN_X_DROPOUT, cfg.RNN_H_DROPOUT, True)
        loss = bucket.cal_loss(vectors, masks, indexes, True, accelerate=True,
                               dropout_x=cfg.RNN_DEC_X_DROPOUT, dropout_h=cfg.RNN_DEC_H_DROPOUT,
                               ccg_dropout=cfg.CCG_DROPOUT, mlp_dropout=cfg.MLP_DROP)
        valid_loss.append(loss.value())
        loss.backward()
        trainer.learning_rate = cfg.LEARNING_RATE * cfg.LR_DECAY ** (cnt_iter / cfg.LR_ANNEAL)
        trainer.update()

        if cnt_iter % cfg.VALID_ITER: continue
        avg_loss = np.sum(valid_loss) / len(valid_loss)
        valid_loss = []
        logger.info("")
        logger.info("Iter: %d-%d, Avg_loss: %s" % (cnt_iter / cfg.VALID_ITER, cnt_iter, avg_loss))
        valid_batch = dev_datasets.get_batches(cfg.TEST_BATCH_SIZE, True, cmp, False)
        good = 0
        total = 0
        for indexes, masks in valid_batch:
            dy.renew_cg()
            vectors = token_repre(indexes, False)
            vectors = encoder(vectors, None, cfg.RNN_X_DROPOUT, cfg.RNN_H_DROPOUT, False)
            good_batch, total_batch = bucket.cal_loss(vectors, masks, indexes, False,
                                                      accelerate=True)
            good += good_batch
            total += total_batch
        acc = good * 1.0 / total
        # print(str(cnt_iter/cfg.VALID_ITER) + "\t genera: " + str(acc))
        logger.info('Results: %s' % acc)
        if acc > BEST_DEV_ACC:
            dy.save(cfg.BEST_MODEL, [token_repre, encoder, decoder])
            with open(cfg.VOCAB_FILE, 'wb') as fp:
                pickle.dump(vocab, fp)
            BEST_DEV_ACC = acc
            # print("best: " + str(BEST_DEV_ACC))
            logger.info('Best Results: %s' % acc)


if __name__ == '__main__':
    main()
