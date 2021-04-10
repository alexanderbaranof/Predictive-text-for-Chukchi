from __future__ import print_function

import sys
import pickle



import argparse
import os
from six.moves import cPickle


from six import text_type

import tensorflow as tf
from model import Model

from tqdm import tqdm

# Hits is number of times we get a prediction right
hits = 0
# The number of tokens
n_tokens = 0


def sample(prime):

    tf.reset_default_graph()

    n = 47
    sample = 2
    save_dir = './char-rnn/save/'

    num_words = len(prime.split(' '))

    flag_first = False

    if prime == '#':
        flag_first = True
        prime = ''

    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    # Use most frequent char if no prime is given
    if prime == '':
        prime = chars[0]
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            data = model.sample(sess, chars, vocab, n, prime,
                                sample).encode('utf-8')
            #print('было', prime, file=sys.stderr)
            #print('предсказали', data.decode("utf-8"), file=sys.stderr)
            if flag_first:
                return data.decode("utf-8").split(' ')[0]
            else:
                return data.decode("utf-8").split(' ')[num_words]


# For each of the lines in the input
for line in tqdm(sys.stdin.readlines()):
    # Split into two columns
    row = line.strip().split('\t')
    # Our tokens are in column one, split by space
    tokens = row[0].split(' ')
    # The test tokens are the beginning of sentence symbol + the list of tokens
    tst_tokens = ['#'] + tokens
    # Increment the number of tokens by the length of the list containing the tokens
    n_tokens += len(tokens)

    # This is our output
    output = []

    # For each of the tokens in the "tst_tokens" list (e.g. the list + the beginning of sentence symbol)
    for i in range(len(tst_tokens)-1):
        first = tst_tokens[i]  # First token in bigram
        second = tst_tokens[i+1]  # Second token in bigram
        # If we find the first token in the bigrams dict
    #

        if i == 0:
            #print(i, ' '.join(tst_tokens[:i+1]), file=sys.stderr)
            predicted_second = sample(' '.join(tst_tokens[:i+1]))
        else:
            #print(i, ' '.join(tst_tokens[1:i+1]), file=sys.stderr)
            predicted_second = sample(' '.join(tst_tokens[1:i+1]))

        #print(i, ' '.join(output),  file=sys.stderr)
        #print(i, 'предсказали', predicted_second, file=sys.stderr)


        # predicted_second = sample(' '.join(tst_tokens[:i+1]))

        if predicted_second == second:
            # We add this whole token to the output
            # e.g. a single click on a prediction
            output.append(predicted_second)
            # Increment the number of hits by 1
            hits += 1
        else:
            # Otherwise we add each individual character to the output
            # e.g. writing out each of the individual clicks
            output += [c for c in second]

        output.append('_')

    # Print out our input and the predicted sequence of keypresses
    print('%s\t%s' % (row[0], ' '.join(output)))

print('Hits:', hits, '; Tokens:', n_tokens, file=sys.stderr)
