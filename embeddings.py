# os and sys
import os
import sys
# numpy
import numpy as np
import bcolz
import pickle
# Paragraph length script to determine top vocabulary
import analyse_paragraph_lengths as apl


# Note: we use a cased version of Glove
# Followed the following approach
# https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

GLOVE_FILE = 'F:\\1QuestionAnswering\\glove\\glove.840B.300d.txt'
GLOVE_PATH = 'F:\\1QuestionAnswering\\glove\\' # todo: join these with os.join
DATASET_PATH = "F:\\1QuestionAnswering\\preprocessed_files\\outputs\\searchqa_test.pkl"


#TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 100 # todo: check out typical length of candiates to decide on this, not specified in paper
VOCAB_SIZE = 1000 # todo: decide on vocab size, not specified by paper
MAX_GLOVE_RETRIEVAL_SIZE = 1000 #only used for debugging, so we do not crate the full 2.4m entries during traing
EMBEDDING_DIM = 300 # 300 dimensions as specified by paper


def build_glove_dict()->dict:
    '''
    Builds a dictionary that maps words to embedding vectors
    :return:
    '''
    print("Building and pickling glove vectors.")
    # List of words that appear in the Glove file
    words = []
    # Index
    start_index = 0
    # Word to index mapping
    word2idx = {}
    # This will be the file where we store our vectors, words and indexes are stored in separate
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{GLOVE_PATH}\\840B.300d.dat', mode='w')

    with open(GLOVE_FILE, 'rb') as f:
        # todo: stop after vocab size is reached
        for idx in range(start_index, MAX_GLOVE_RETRIEVAL_SIZE):
            line = next(f).decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
            if idx%10000 == 0 and idx > 0:
                print("Read in {} glove vectors.".format(idx))


    vectors = bcolz.carray(vectors[1:].reshape((-1, EMBEDDING_DIM)), rootdir=f'{GLOVE_PATH}\\840B.300d.dat', mode='w')
    # Write vectors to disk
    vectors.flush()
    # Save word list to file
    pickle.dump(words, open(f'{GLOVE_PATH}\\840B.300d_words.pkl', 'wb'))
    # save index list to file
    pickle.dump(word2idx, open(f'{GLOVE_PATH}\\840B.300d_idx.pkl', 'wb'))

    # Build vocab that maps index to word
    glove = {w: vectors[word2idx[w]] for w in words}
    # test
    assert len(glove) == MAX_GLOVE_RETRIEVAL_SIZE, "Vocabsize does not match length of Vocabulary Dictionary."
    return glove

def load_pickled_glove():
    # Load vectors, words and words to index files
    vectors = bcolz.open(f'{GLOVE_PATH}\\840B.300d.dat')[:]
    words = pickle.load(open(f'{GLOVE_PATH}\\840B.300d_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{GLOVE_PATH}\\840B.300d_idx.pkl', 'rb'))
    # Build vocab that maps index to word
    glove = {w: vectors[word2idx[w]] for w in words}
    return glove




def make_emedding_matrix(glove_dict, target_vocab):
    '''
    Returns a matrix of size VOCAB_SIZE*EMBEDDING_DIM
    :param glove_dict:
    :param target_vocab:
    :return:
    '''
    print("making embedding matrix.")
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, EMBEDDING_DIM))
    idx_2_word = {}
    word_2_idx = {}
    words_found = 0

    for i, word in enumerate(target_vocab):
        try:
            weights_matrix[i] = glove_dict[word]
            words_found += 1
        # If the glove dictionary does not not contain the word, add random vector
        except KeyError:
            #todo: figure out what scale does
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM, ))
        idx_2_word[i] = word
        word_2_idx[word] = i
    print('{} of the {} words in the quasar/searchqa set were found in the glove set'.format(words_found, matrix_len))
    return weights_matrix, idx_2_word, word_2_idx


def main(process_glove=False):
    if process_glove:
        glove = build_glove_dict()
    else:
        glove = load_pickled_glove()
    pass

    # Retrieve most used vocabulary words from the dataset file
    _, vocabulary = apl.count_length_values(DATASET_PATH, type='searchqa')
    # todo: note that this vocabulary was simply split with the split function
    # a tokenizer function would be much nicer
    # todo: we need some function that tokenizes all contexts properly, gives them
    # while doing that it should track how often each vocabulary item appears
    # the ones appaering most often will be the top_vocabulary
    top_vocabulary = [x[0] for x in vocabulary.most_common(VOCAB_SIZE)]
    emb_mtx, idx_2_word, word_2_idx = make_emedding_matrix(glove_dict=glove, target_vocab=top_vocabulary)
    print(emb_mtx.shape)

if __name__ == "__main__":
    main(process_glove=True)


'''
pytorch embedding layer
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
'''