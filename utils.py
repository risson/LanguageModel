import numpy as np
from numpy import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def generate_big_text(txtfile, encoding='gbk'):
    
    texts = ''

    with open(txtfile, 'r', encoding=encoding) as fopen:
        for i, line in enumerate(fopen):
            
            # use 10000 lines for test
            if i > 10000: 
                break
            texts += line
    
    texts = texts.lower()
    return texts

def generate_sentences(texts, maxlen, tokenizer, mode='train'):
    charindex = tokenizer.word_index
    seq_X = []
    y = []
    for i in np.arange(1, len(texts)):
        if i-maxlen < 0:
            sent_X = texts[:i]
        else:
            sent_X = texts[i-maxlen:i]
        sent_y = charindex[texts[i]]
        seq_X.append(sent_X)
        y.append(sent_y)
    X = tokenizer.texts_to_sequences(seq_X)
    X = pad_sequences(X, maxlen=maxlen, padding='pre', truncating='post', value=0)

    return X, y

def negative_sample_array(tokenizer, negative_array_size=1e7):
    """
    """
    word_count = tokenizer.word_counts
    word_count_refine = dict((_[0],_[1]**0.75) for _ in word_count.items())
    refine_cnt = sum(_[1] for _ in word_count_refine.items())
    word_freq = dict((_[0], _[1]/refine_cnt) for _ in word_count_refine.items())       

    negative_array = []

    i = 0    
    for word, freq in word_freq.items():
        if i >= negative_array_size: break
        word_num = round(freq*negative_array_size)
        for _ in range(word_num):
            if i >= negative_array_size: continue
            word_idx = tokenizer.word_index[word]
            negative_array.append(word_idx)
            i += 1
    
    random.seed(1)
    random.shuffle(negative_array)
    return negative_array

if __name__ == '__main__':
    texts = generate_big_text('test.txt','gbk')
    tok = Tokenizer(char_level=True, oov_token='<UNK>')
    tok.fit_on_texts([texts])
    ci = tok.word_index
     
    a,b = generate_sentences(texts, 5, tok, 'train')

    nsp = negative_sample_array(tok, 10000)

    print(nsp[:len(a)])