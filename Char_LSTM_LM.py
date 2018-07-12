import numpy as np
import keras as K
from keras.layers import LSTM, Dense, Activation
from keras import Sequential
from keras.preprocessing.text import Tokenizer
import sys
from keras.callbacks import LambdaCallback
import utils


class LMCharLSTM():

    def __init__(self, txtfile):
        
        self._txtfile = txtfile
        self.maxlen = 128

        texts = utils.generate_big_text(txtfile)
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts([texts])
        self.charindex = tokenizer.word_index
        self.indexchar = dict(zip(_[1],_[0]) for _ in self.charindex.items())
        self.diversity = [0.2, 0.4, 0.6, 0.8]

        self._model = self.__init_model__()

    
    def __init_model(self):
        model = Sequential()
        model.add(LSTM(512, activation='tanh', return_sequences=True))
        model.add(LSTM(256, activation='tanh', return_sequences=False))
        model.add(Dense(len(self.chardict)))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam',loss='categorical_crossentropy')
        return model 

    def train(self, epochs=50, batch_size=256):

        def epoch_end_print(epoch, logs):
            print()
            print('-----Generate text for epoch {}: -----'.format(epoch))
            texts = utils.generate_big_text(txtfile)

            i = np.random.randint(len(texts)-self.maxlen)
            sentence = texts[i:i+self.maxlen]
            print('Seed text: ')
            print('"' + sentence + '"')
            print('Generated text: ')
            
            for _ in range(300):
                
                reader = utils.sentence_reader(sentence, self.maxlen, self.charindex, 'test')
                vec = next(reader)
                preds = self._model.predict(np.array(vec), verbose=0)[0]
                nextcharindex = np.random.choice(len(self.charindex), p=preds)
                nextchar = self.indexchar[nextcharindex]
                sentence = sentence[1:]+nextchar

                sys.stdout.write(nextchar)
                sys.stdout.flush()

            print()
                
        #----- Read one sentence at a time
        for epoch in range(epochs):
            texts = utils.generate_big_text(self._txtfile)
            del texts
            X_batch = []
            y_batch = []
            loss_list = []
            batch_id = 0

            for sentence, nextchar in utils.sentence_reader(texts, self.maxlen, self.charindex, 'train'):    
                batch_id += 1               
                
                if batch_id%(batch_size*1000)==0:
                    print('{} - Epoch {}: Batch_id - {}: Loss = {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                          epoch, batch_id, np.mean(loss_list)))
                    loss_list = []

                if batch_id%batch_size==0:
                    
                    loss = self._model.train_on_batch(np.array(X_batch),np.array(y_batch))
                    loss_list.append(loss)
                    X_batch = []
                    y_batch = []
                
                X_batch.append(sentence)
                y_batch.append(nextchar)
            
            epoch_end_print(epoch,None)

        #----- Read all data into memory:
        reader = utils.sentence_reader(self._txtfile, self.maxlen, self.charindex)
        X = []
        y = []
        for sentence, nextchar in reader():
            X.append(sentence)
            y.append(nextchar)

        print_callback = LambdaCallback(on_epoch_end=epoch_end_print)
        self._model.fit(X, y,
                        epochs=50,
                        batch_size=256,
                        callbacks=[print_callback])


if __name__ == '__main__':
    text_gen = LMCharLSTM()
    text_gen.train()
