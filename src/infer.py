import os 
import tensorflow as tf
from data_load import get_batch_for_inference, vocabs
from model import Transformer
from hparams import Hparams
from utils import postprocess
tf.logging.set_verbosity(tf.logging.ERROR)

class Epitope2Tcr:
    def __init__(self, ckpt_path=''):
        hparams = Hparams()
        parser = hparams.parser
        
        if ckpt_path:
            self.hp = parser.parse_args(
                ['--ckpt', ckpt_path])
        else:
            self.hp = parser.parse_args()

        self.transformer = Transformer(self.hp)
        #self._restore_trained_model()

    def infer(self, epitopes):
        epitopes = [e for e in epitopes if self._is_valid(e)]
        test_batches, num_test_batches, num_test_samples = \
            get_batch_for_inference(epitopes, self.hp.test_batch_size)
        
        iter = tf.data.Iterator.from_structure(test_batches.output_types,
                                               test_batches.output_shapes)
        xs, ys = iter.get_next()
        m = Transformer(self.hp)
        y_hat, _ = m.eval(xs, ys)

        self._restore_trained_model()
        output_sequence = []
        for _ in range(num_test_batches):
            seq = sess.run(y_hat)
            output_sequence.extend(seq.tolist())
        output_sequence = postprocess(
            output_sequence, self.transformer.idx2token)

        return output_sequence


    def _is_valid(self, seq, vocabs=vocabs):
        return set([a for a in seq]) in set(vocabs) and len(seq) > 0

    def _restore_trained_model(self):
        with tf.Session() as sess:
            ckpt_ = tf.train.latest_checkpoint(self.hp.ckpt)
            ckpt = self.hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir
            saver = tf.train.Saver()

            saver.restore(sess, ckpt)
        
