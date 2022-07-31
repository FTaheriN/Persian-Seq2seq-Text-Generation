# generate sequnce

import numpy as np
from random import randint
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from cleanText import readFile


with open('tokenizer.txt', 'rb') as handle:
  word_dic = pickle.loads(handle.read())
 
# generate a sequence from a language model
def generate_seq(model, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
	# generate a fixed number of words
    for _ in range(n_words):
	    # encode the text as integer
        sentence = in_text.split()
        encoded = []
        for token in sentence:
            encoded.append(word_dic[token])
	    # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
	    # predict probabilities for each word
        predict=model.predict(encoded) 
        yhat=np.argmax(predict,axis=1)
	    # map predicted word index to word
        out_word = ''
        for word, index in word_dic.items():
            if index == yhat:
                out_word = word
                break
	    # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

def test():
    # load cleaned text sequences
    in_filename = 'hafez_seq.txt'
    doc = readFile(in_filename)
    lines = doc.split('\n')
    seq_length = len(lines[0].split()) - 1
    
    # load the model
    model = load_model('/Model/keras_model.h5')
    
    # select a seed text
    seed_text = lines[randint(0,len(lines))]
    print(seed_text + '\n')
    
    # generate new text
    generated = generate_seq(model, seq_length, seed_text, 8)
    print(generated)
    
test()