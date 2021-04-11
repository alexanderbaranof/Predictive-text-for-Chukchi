import sys

import pandas as pd


path_for_train_file = sys.argv[1]


dict_index = 0
result_dict = dict()


for i, line in enumerate(open(path_for_train_file).readlines()):
    row = line.strip().split('\t')
    tokens = row[0].split(' ')
    for id_tok, tok in enumerate(tokens):
        if id_tok > 0:
            #print('type', id_tok+1, 'text', ' '.join(tokens[:id_tok]), 'label', tok)
            result_dict[dict_index] = {'type': id_tok+1, 'text': ' '.join(tokens[:id_tok]), 'label': tok}
            dict_index += 1


    # if i > 5:
    #    break


df = pd.DataFrame().from_dict(result_dict, orient='index')
df.to_csv('../data/df_for_train_classifire.csv')    


#f = open('./data/dataset_for_char_rnn.txt', 'w')
#f.write(result_text)

