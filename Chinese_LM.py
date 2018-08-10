import numpy as np
from keras import backend as K
from keras.layers import LSTM, Dense, Activation, Masking, Input
from keras import Sequential
from keras.preprocessing.text import Tokenizer
import sys
from keras.callbacks import LambdaCallback
import utils


class ChnLM():

    def __init__(self, txtfile):
        
        self._txtfile = txtfile
        self.TIMESTEPS = 20
 
        texts = utils.generate_big_text(txtfile)
        tokenizer = Tokenizer(char_level=True, oov_token='<UNK>')
        tokenizer.fit_on_texts([texts])
        
        self.charindex = tokenizer.word_index
        self.vocabsize = len(self.charindex)
        self.diversity = [0.2, 0.4, 0.6, 0.8]

        self._model = self.__init_model()

    
    def __init_model(self):
        embedding_mtrx = self.__get_embedding_wgt()
        
        input_left = Input(shape=(self.TIMESTEPS,), name='input_left')
        masking = Masking(mask_value=0, name='masking')(input_left)
        

        return model 

    def __get_embedding_wgt(self):
        embedding_idx = dict()
        with open('', 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_idx[word] = coefs
        
        embedding_mtrx = zeros((self.vocabsize, 300))
        for word, i in self.charindex.items():
            embedding_vec = embedding_idx.get(word)
            if embedding_vec is not None:
                embedding_mtrx[i] = embedding_vec

        return embedding_mtrx


    def train(self, epochs=50, batch_size=256):

        texts = utils.generate_big_text(self._txtfile)

        def epoch_end_print(epoch, logs):
            print()
            print('-----Generate text for epoch {}: -----'.format(epoch))
            i = np.random.randint(len(texts)-self.maxlen-1)
            sentence = texts[i:i+self.maxlen]
            print('Seed text: ')
            print('"' + sentence + '"')
            print('Generated text: ')
            sys.stdout.write(sentence)

            for _ in range(30):
                
                reader = utils.sentence_reader(sentence, self.maxlen, self.charindex, 'test')
                vec = next(reader())
                preds = self._model.predict(np.array([vec]), verbose=0)[0]
                nextcharindex = np.random.choice(len(self.charindex), p=preds)
                nextchar = self.indexchar[nextcharindex]
                sentence = sentence[1:]+nextchar

                sys.stdout.write(nextchar)
                sys.stdout.flush()

            print()
        
        reader = utils.sentence_reader(texts, self.maxlen, self.charindex, 'train')

        
        X = []
        y = []
        
        for sentence, nextchar in reader():
            X.append(sentence)
            y.append(nextchar)
 
        print_callback = LambdaCallback(on_epoch_end=epoch_end_print)
        self._model.fit(np.array(X), np.array(y),
                        epochs=50,
                        batch_size=256,
                        callbacks=[print_callback])


if __name__ == '__main__':
    txtfile = 'reviews_Electronics_5.txt'
    text_gen = LMCharLSTM(txtfile)
    text_gen.train()
