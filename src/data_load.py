import pandas as pd
from utils import calc_num_batches
import tensorflow as tf

vocabs = [a for a in 'ACDEFGHIKLMNOPQRSTVWXY']

def load_vocab(vocabs=vocabs):
    '''
    idx <-> token mapping
    These mappings are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>
    '''
    token2idx = {token: idx+4 for idx, token in enumerate(vocabs)}
    token2idx.update({
        '<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3
    })
    idx2token = {idx: token for token, idx in token2idx.items()}

    return token2idx, idx2token


def pad(token_list, max_len):
    """
    tf is supposed to take care of this but I did this manually anyways.
    """
    return token_list + ['<pad>'] * (max_len - len(token_list))


def encode(seq, type, dict, max_len):
    '''
    seq: string. Amino acid sequence
    type: "epitope" (source), "cdr3": (target)
    dict: token2idx
    '''

    if type == 'epitope': tokens = [c for c in seq] + ['</s>']
    else: tokens = ['<s>'] + [c for c in seq] + ['</s>']

    return [dict.get(t, dict['<unk>']) for t in tokens]


def generator_fn(epitopes, cdr3s, vocabs):
    token2idx, _ = load_vocab(vocabs)

    max_len_e = len(max(epitopes, key=len))
    max_len_c = len(max(cdr3s, key=len))

    for epitope, cdr3 in zip(epitopes, cdr3s):
        x  = encode(epitope, 'epitope', token2idx, max_len_e)
        y = encode(cdr3, 'cdr3', token2idx, max_len_c)
        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)

        yield (x, x_seqlen, epitope), (decoder_input, y, y_seqlen, cdr3)


def input_fn(source_sents, target_sents, vocabs, batch_size, shuffle=True):
    """
    Input
    source_sents: list of source sents
    target_sents: list of target sents
    vocabs: list of vocabulary
    batch_size: int
    shuffle: bool

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlen: int32 tensor. (N,)
        sent1: string tensor.  (N,)

    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N,)
        sent2: str tensor. (N,)
    """

    shapes = (([None], (), ()),
              ([None], [None], (), ()))
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.int32, tf.string))
    paddings = ((0, 0, ''),
                (0, 0, 0, ''))

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(source_sents, target_sents, vocabs))

    if shuffle:
        dataset = dataset.shuffle(128 * batch_size)

    dataset = dataset.repeat()
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset


def get_batch(csv_path, batch_size, vocabs=vocabs, shuffle=True):
    df = pd.read_csv(csv_path)
    epitopes = df.epitope.apply(str).tolist()
    cdr3s = df.cdr3.apply(str).tolist()
    batches = input_fn(epitopes, cdr3s, vocabs, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(epitopes), batch_size)

    return batches, num_batches, len(epitopes)


def get_batch_for_inference(epitopes, batch_size, vocabs=vocabs, shuffle=True):
    """
    epitopes: list of str
    """
    batches = input_fn(epitopes, epitopes, vocabs, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(epitopes), batch_size)

    return batches, num_batches, len(epitopes)
