# Predictive-text-for-Chukchi

### Project requirements:
* tensorflow==1.13.0rc1 (for Char RNN)
* tensorflow==2.4.1 (for other methods)
* keras==2.1.5
* gensim==3.7.3
* tqdm
* pandas

### Position based statistics
For run position-based-predictions:
```
python3 ./position-based-predictions/train_predict_position_based.py < ./data/dev.tsv > ./position-based-predictions/output.tsv
python3 evaluate.py ./data/dev.tsv ./position-based-predictions/output.tsv
```

The results of Position-based-predictions:
```
Characters: 37897
Tokens: 8788
Clicks: 37712
Clicks/Token: 4.291306326809285
Clicks/Character: 0.9951183470987149
```

For run position-based-predictions (test):
```
python3 ./position-based-predictions/train_predict_position_based.py < ./data/test/test.tsv > ./position-based-predictions/output_test.tsv
python3 evaluate.py ./data/test/test.tsv ./position-based-predictions/output_test.tsv
```

The results of Position-based-predictions (test):
```
Characters: 37927
Tokens: 8374
Clicks: 37751
Clicks/Token: 4.508120372581801
Clicks/Character: 0.9953595064202283
```



### Position based with bigrams statistics 

For run position-based-predictions-with-bigrams:
```
python3 ./position-based-predictions-with-bigrams/train_predict_position_based_bigrams.py < ./data/dev.tsv > ./position-based-predictions-with-bigrams/output.tsv
python3 evaluate.py ./data/dev.tsv ./position-based-predictions-with-bigrams/output.tsv
```

The results of Position-based-predictions-with_bigrams:
```
Characters: 37897
Tokens: 8788
Clicks: 37693
Clicks/Token: 4.289144287664998
Clicks/Character: 0.9946169881520964
```

For run position-based-predictions-with-bigrams (test):
```
python3 ./position-based-predictions-with-bigrams/train_predict_position_based_bigrams.py < ./data/test/test.tsv > ./position-based-predictions-with-bigrams/output_test.tsv
python3 evaluate.py ./data/test/test.tsv ./position-based-predictions-with-bigrams/output_test.tsv
```

The results of Position-based-predictions-with_bigrams (test):
```
Characters: 37927
Tokens: 8374
Clicks: 37682
Clicks/Token: 4.49988058275615
Clicks/Character: 0.9935402220054315
```


### Char RNN (LSTM)
For run char-rnn:
```
python3 ./char-rnn/predict_next_word_char_rnn.py < ./data/dev.tsv > ./char-rnn/output.tsv
python3 evaluate.py ./data/dev.tsv ./char-rnn/output.tsv
```


The results of char-rnn (predict whole word)
```
Characters: 37897
Tokens: 8788
Clicks: 37848
Clicks/Token: 4.306781975421028
Clicks/Character: 0.998707021663984
```


### Char RNN (LSTM) (Next character predict)
For run char-rnn:
```
 python3 ./char-rnn/predict_next_char_char_rnn.py < ./data/dev.tsv > ./char-rnn/output_by_char.tsv
 python3 evaluate.py ./data/dev.tsv ./char-rnn/output_by_char.tsv
```

The results of char-rnn (predict char by char)
```
Characters: 37897
Tokens: 8788
Clicks: 37727
Clicks/Token: 4.293013199817934
Clicks/Character: 0.9955141567934137
```


### Word level LSTM + FastText for Classification task (like one shot-learning language model)
For run this method see `language-model-like-classification-task/train_and_predict_with_fasttext.ipynb`

The results of LSTM + FastText
```
Characters: 37897
Tokens: 8788
Clicks: 37759
Clicks/Token: 4.2966545289030496
Clicks/Character: 0.9963585508087711
```

The results of LSTM + FastText (test)
```
Characters: 37927
Tokens: 8374
Clicks: 37834
Clicks/Token: 4.518032003821352
Clicks/Character: 0.9975479210061434
```


### Token-level-language-model (CNN)
For run this method see `token-level-language-model/train_predict.ipynb`

The results of Token-level-language-model (CNN)
```
Characters: 37897
Tokens: 8788
Clicks: 37752
Clicks/Token: 4.295857988165681
Clicks/Character: 0.9961738396179117
```

The results of Token-level-language-model (CNN) (test)
```
Characters: 37927
Tokens: 8374
Clicks: 37911
Clicks/Token: 4.527227131597803
Clicks/Character: 0.9995781369472935
```