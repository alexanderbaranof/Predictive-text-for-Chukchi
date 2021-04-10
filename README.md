# Predictive-text-for-Chukchi

### Project requirements:
* tensorflow==1.13.0rc1
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
