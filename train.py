# train the model on the dataset

import re
import numpy as np
import pickle

from buildModel import buildModel


from cleanText import readFile

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def train():
    # load
    in_filename = '/hafez_seq.txt'
    doc = readFile(in_filename)
    lines = doc.split('\n')

    # Assigning each token an integer and converting word sequnces to number sequence
    word_idx = 1
    word_dic = {}
    doc_seq = []
    sentence = []
    for line in lines:
        sentence.clear()
        for token in line.split():
            if (token not in word_dic):
                word_dic[token] = word_idx
                word_idx += 1
            sentence.append(word_dic[token])
        doc_seq.append(sentence[:])

    print("number of lines: ", len(doc_seq))
    print("number of vocabulary: ", len(word_dic))

    vocab_size = len(word_dic) + 1
    doc_seq = np.array(doc_seq)

    # prepare data for training 
    X, y = doc_seq[:,:-1], doc_seq[:,-1]
    # one-hot encode the vocabulary output
    y = to_categorical(y, vocab_size)

    seq_length = X.shape[1]

    # define model
    model = buildModel(vocab_size, seq_length)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X, y, batch_size=128, epochs=100)
    
    # save the model to file
    model.save('/Model/keras_model.h5')
    # save the tokenizer
    with open('tokenizer.txt', 'wb') as handle:
      pickle.dump(word_dic, handle)
  
  
train()


