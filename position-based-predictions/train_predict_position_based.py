import sys, pickle
import copy
import pandas as pd
from tqdm import tqdm

training_file = './data/train.tsv'

# Hits is number of times we get a prediction right
hits = 0
# The number of tokens
n_tokens = 0


result_dict = dict()

for i, line in enumerate(open(training_file).readlines()):
    row = line.strip().split('\t')
    tokens = row[0]
    for id_tok, tok in enumerate(tokens.split(' ')):
        if id_tok in result_dict:
            result_dict[id_tok].append(tok)
        else:
            result_dict[id_tok] = list()
            result_dict[id_tok].append(tok)



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

        
        predicted_second = pd.Series(result_dict[i]).value_counts()[:1].index[0]

        #print(i, predicted_second,  file=sys.stderr)
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