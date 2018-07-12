import numpy as np
from keras.preprocessing.text import Tokenizer

def generate_big_text(txtfile):
    
    texts = ''

    with open(txtfile, 'r', encoding='utf-8') as fopen:
        for i, line in enumerate(fopen):
            
            # use 10000 lines for test
            if i > 10000: 
                break
            texts += line
    
    texts = texts.lower()
    return texts

def sentence_reader(texts, maxlen, charindex, mode='train'):
    
    for i in np.arange(0, len(texts), maxlen):
        if (i+maxlen>=len(texts)) and mode=='train':
            break
        text = texts[i:i+maxlen]
        sent_X = []
        for char in text:            
            
            char_idx = charindex[char]
            feat_list = [1 if i==char_idx else 0 for i in range(len(charindex))]
            sent_X.append(feat_list)

        if mode=='test':
            yield(sent_X)
        else:
            sent_y = charindex[texts[i+maxlen]]
            yield (sent_X, sent_y)

if __name__ == '__main__':
    texts = generate_big_text('C:/Users/risson.yao/word2vec/testpage.txt')
    tok = Tokenizer(char_level=True)
    tok.fit_on_texts([texts])
    ci = tok.word_index
    print(ci)

    print(texts[:10])
    
    reader = sentence_reader(texts[:10],10,ci,'test')

    a = next(reader)
        
    print(a)