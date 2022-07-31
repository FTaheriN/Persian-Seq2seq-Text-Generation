# tokenize and clean text 
# save text into fixed length lines

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.fa import Persian

import re


farsi = Persian()
fa_Tokenizer = Tokenizer(farsi.vocab)

def readFile(filename):
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def tokenizeSentence(sentence):
    return  [word.text for word in 
             fa_Tokenizer(re.sub(r"\s+"," ",re.sub(r"[\.\!\؟\,\'\"\n+]"," ",sentence)).strip())]

# save tokens to file, eight words per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


def main():
    filename = "hafez_norm.txt"
    doc = readFile(filename)
    data = tokenizeSentence(doc)
    print("tokenized data: ", data[:25])

    # Making sentences with tokens with the desired length
    length = 8
    sequences = list()
    for i in range(length, len(data)):
        # select sequence of tokens
        seq = data[i-length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))

    # save sequences to file
    out_filename = 'hafez_seq.txt'
    save_doc(sequences, out_filename)
    
main()
