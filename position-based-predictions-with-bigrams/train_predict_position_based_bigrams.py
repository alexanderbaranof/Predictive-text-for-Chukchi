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

    splited_tokens = tokens.split(' ')

    for id_tok in range(len(splited_tokens)-1):
        tok = splited_tokens[id_tok]
        next_tok = splited_tokens[id_tok+1]
        tok = f'{tok} {next_tok}'
        bigram_id = str(id_tok +1) + '_' + str(id_tok + 2)
        if bigram_id in result_dict:
            result_dict[bigram_id].append(tok)
        else:
            result_dict[bigram_id] = list()
            result_dict[bigram_id].append(tok)




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

        bigram_id = str(i) + '_' + str(i + 1)

        if bigram_id == '0_1':
            # Это начало, а значит надо выбрать просто самое популярное первое слово
            all_first_words = list()
            for bigram in result_dict['1_2']:
                all_first_words.append(bigram.split(' ')[0])
            all_first_words = pd.Series(all_first_words)
            predicted_second = all_first_words.value_counts().index.tolist()[0]
        else:
            all_allowed_bigrams = list()
            for bigram in result_dict[bigram_id]:
                if bigram.split(' ')[0] == first:
                    all_allowed_bigrams.append(bigram)
            if len(all_allowed_bigrams) > 0:
                all_allowed_bigrams = pd.Series(all_allowed_bigrams)
                #print('tut', all_allowed_bigrams.value_counts().index.tolist()[0].split(' ')[1], file=sys.stderr)
                predicted_second = all_allowed_bigrams.value_counts().index.tolist()[0].split(' ')[1]
            else:
                all_allowed_bigrams = list()
                for k in result_dict:
                    for bigram in result_dict[k]:
                        if bigram.split(' ')[0] == first:
                            all_allowed_bigrams.append(bigram)
                
                if len(all_allowed_bigrams) > 0:
                    all_allowed_bigrams = pd.Series(all_allowed_bigrams)
                    predicted_second = all_allowed_bigrams.value_counts().index.tolist()[0].split(' ')[1]
                else:
                    predicted_second = first

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