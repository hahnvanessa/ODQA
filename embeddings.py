# os and sys
import os
import sys
# numpy
import numpy as np
# keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant


# Note: we use a cased version of Glove
# Note: following this tutorial https://keras.io/examples/pretrained_word_embeddings/

#
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
# Text as ?
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000 # todo: check out typical length of candiates to decide on this, not specified in paper
MAX_NUM_WORDS = 20000 # todo: decide on vocab size, not specified by paper
EMBEDDING_DIM = 300 # 300 dimensions as specified by paper



# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

