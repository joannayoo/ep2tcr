# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Inference
'''

import os

import tensorflow as tf

from data_load import get_batch_for_inference, vocabs
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, calc_bleu, postprocess, load_hparams
import logging
logging.basicConfig(level=logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)



print("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)


epitopes = ['YMPTTIIAA',
            'TVYGFCLL',
            'TPQDLNTML',
            'EAAGIGILTV',
            'ASNENMETM',
            'GILGFVFTL',
            'ASNENMETM',
            'HPKVSSEVHI',
            'LSLRNPILV']

print("# Prepare test batches")
test_batches, num_test_batches, num_test_samples  = get_batch_for_inference(
    epitopes, hp.test_batch_size, shuffle=False)
iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
xs, ys = iter.get_next()

test_init_op = iter.make_initializer(test_batches)

print("# Load model")
m = Transformer(hp)
y_hat, _ = m.eval(xs, ys)

print("# Session")
with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)

    sess.run(test_init_op)

    print("# get hypotheses")
    hypotheses = get_hypotheses(num_test_batches, num_test_samples, sess, y_hat, m.idx2token)
    

    """
    print("# write results")
    model_output = ckpt.split("/")[-1]
    if not os.path.exists(hp.testdir): os.makedirs(hp.testdir)
    translation = os.path.join(hp.testdir, model_output)
    with open(translation, 'w') as fout:
        fout.write("\n".join(hypotheses))

    print("# calc bleu score and append it to translation")
    calc_bleu(hp.test2, translation)
    """

print(hypotheses)
