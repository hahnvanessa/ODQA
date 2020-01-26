# os and sys
import os
import sys
# numpy
import numpy as np
import bcolz
import pickle
# Paragraph length script to determine top vocabulary
import analyse_paragraph_lengths as apl
from collections import Counter
# tokenization
from tokenizers import (ByteLevelBPETokenizer,
                            BPETokenizer,
                            SentencePieceBPETokenizer,
                            BertWordPieceTokenizer)
import spacy
# Disable spacy components to speed up tokenization
nlp = spacy.load('en_core_web_sm',disable=['tagger', 'parser', 'ner', 'print_info']) # let's start with the small model

# Note: we use a cased version of Glove
# Followed the following approach
# https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

GLOVE_FILE = 'F:\\1QuestionAnswering\\glove\\glove.840B.300d.txt'
GLOVE_PATH = 'F:\\1QuestionAnswering\\glove\\' # todo: join these with os.join
# todo: change paths from test to train
DATASET_PATH_SEARCHQA = "F:\\1QuestionAnswering\\preprocessed_files\\outputs\\searchqa_test.pkl" #pickled
DATASET_PATH_QUASAR = "F:\\1QuestionAnswering\\preprocessed_files\\outputs\\quasar_test_short.pkl" #pickled
# Output Pathes
OUTPUT_PATH_ENCODED = "F:\\1QuestionAnswering\\preprocessed_files\\outputs\\"

#TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 54 # should approximately be the mean sentence length+ 0.5std, searchqa has highest with appr. 54
VOCAB_SIZE = 1000 # todo: decide on vocab size, not specified by paper, good value might be 20000
MAX_GLOVE_RETRIEVAL_SIZE = 1000 #todo: only used for debugging, so we do not crate the full 2.4m entries during traing
EMBEDDING_DIM = 300 # 300 dimensions as specified by paper
PAD_IDENTIFIER = '<PAD>'
UNKNOWN_IDENTIFIER = '<UNK>'

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


def tokenize_context(context) -> list:
    '''
    Tokenizes the given context by using spacy tokenizer.
    :param context:
    :return:
    '''
    # Deactivate spacy components to speed up the process
    # todo: switch to spacy tokenizer (takes longer but is more accurate)
    return context.split() #[token.text for token in nlp(context)]


def tokenize_set(DATASET_PATH, type='quasar'):
    print("Applying tokenization to set: {}".format(type))
    #todo: check if it is efficient and indexwise ok to store all contexts in a simple list, alternative: array
    assert type in ['quasar', 'searchqa'], 'Wrong type specified. Allowed types are "quasar" and "searchqa'
    # Count all tokens
    token_count = Counter()

    # Read in pickled dictionary
    pickle_in = open(DATASET_PATH, "rb")
    corpus_dict = pickle.load(pickle_in)
    i = 0
    # Extract passages from the dictionary
    for question_id, qv in corpus_dict.items():
        corpus_dict[question_id]['tokenized_contexts'] = []
        if type == 'quasar':
            for _, context in qv['contexts']:
                # Note this also counts punctation but it should still return an approximate solution
                tokenized_context = tokenize_context(context)
                corpus_dict[question_id]['tokenized_contexts'].append(tokenized_context)
                token_count.update(tokenized_context)
        else:
            for context in qv['contexts']:
                if context:
                    tokenized_context = tokenize_context(context)
                    corpus_dict[question_id]['tokenized_contexts'].append(tokenized_context)
                    token_count.update(tokenized_context)

        # Delete untokenized contexts to save memory
        del corpus_dict[question_id]['contexts']

        i += 1
        if i%1000 == 0:
            print('Tokenized {} of {} questions in total for set <{}>'.format(i, len(corpus_dict), type))
    return corpus_dict, token_count

def make_emedding_matrix(glove_dict, target_vocab):
    '''
    Returns a matrix of size VOCAB_SIZE*EMBEDDING_DIM.
    Target vocabulary are the top n most occuring words
    in a set.
    :param glove_dict:
    :param target_vocab:
    :return:
    '''
    print("making embedding matrix.")
    matrix_len = len(target_vocab)
    # Weight matrix has vocabulary plus the entries for padding and unknown
    weights_matrix = np.zeros((matrix_len+2, EMBEDDING_DIM))
    idx_2_word = {}
    word_2_idx = {}
    words_found = 0


    # Add padding symbol
    idx_2_word[0] = PAD_IDENTIFIER
    word_2_idx[PAD_IDENTIFIER]  = 0
    pad_vector = np.zeros(shape=(EMBEDDING_DIM,))
    weights_matrix[0] = pad_vector
    # Add unknown word (word that does not appear in the top vocabulary)
    idx_2_word[1] = UNKNOWN_IDENTIFIER
    word_2_idx[UNKNOWN_IDENTIFIER] = 1
    unknown_vector = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
    weights_matrix[1] = unknown_vector


    for i, word in enumerate(target_vocab, start = 2):
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

def encode_pad_context(tokenized_context, word_2_idx):
    '''
    Given a tokenized context, pads or trunctuates the context, applies index on tokens
    and returns a numpy array.
    :param tokenized_context:
    :param word_2_idx:
    :return:
    '''
    encoded_context = []
    # Pad context or trunctuate context
    tokenized_context.extend([PAD_IDENTIFIER] * MAX_SEQUENCE_LENGTH) # always add so min size is always MAX_SEQUENCE_LENGTH + 1
    tokenized_context = tokenized_context[:MAX_SEQUENCE_LENGTH]
    # Index words, identify unkown ones
    for t in tokenized_context:
        # If word in top vocabulary or is a padding token
        if t in word_2_idx:
            encoded_context.append(word_2_idx[t])
        else:
            encoded_context.append(word_2_idx[UNKNOWN_IDENTIFIER])
    return np.array(encoded_context)

def encode_corpus_dict(corpus_dict, word_2_idx) -> dict:
    '''
    Applies encoding to a corpus dict that already contains tokenized contexts.
    Adds a list of encoded context to the particular question_id dict.
    :param corpus_dict: Corpus dict of either searchqa or quasar
    :param word_2_idx: Mapping of words to indexes
    :return:
    '''
    i = 0
    for question_id, qv in corpus_dict.items():
        corpus_dict[question_id]['encoded_contexts'] = []
        for tokenized_context in qv['tokenized_contexts']:
            encoded_context = encode_pad_context(tokenized_context,word_2_idx)
            corpus_dict[question_id]['encoded_contexts'].append(encoded_context)
        i += 1
        if i % 1000 == 0:
            print('Encoded {} of {} questions in total'.format(i, len(corpus_dict)))
        # Delete tokenized entries to save memory
        del corpus_dict[question_id]['tokenized_contexts']

    return corpus_dict

def main(process_glove=False):
    # Specify whether the original Glove file should be processed or a
    # already pickled version of Glove should be loaded
    if process_glove:
        glove = build_glove_dict()
    else:
        glove = load_pickled_glove()

    # Tokenization
    # 1. Retrieve most used vocabulary words from searchqa and quasar
    # 2. Tokenize contexts but do not apply padding yet, tokenized contexts stored in corpus_dict
    searchqa_tok_corpus_dict, searchqa_token_count = tokenize_set(DATASET_PATH_SEARCHQA, type='searchqa')
    quasar_tok_corpus_dict, quasar_token_count = tokenize_set(DATASET_PATH_QUASAR, type='quasar')
    # 3. Combine the top vocabularies from both sets
    total_token_count = searchqa_token_count + quasar_token_count
    top_vocabulary = [x[0] for x in total_token_count.most_common(VOCAB_SIZE)]
    del quasar_token_count # save memory
    del searchqa_token_count # save memory

    # Create an embedding matrix
    emb_mtx, idx_2_word, word_2_idx = make_emedding_matrix(glove_dict=glove, target_vocab=top_vocabulary)

    # Test encoding
    print(encode_pad_context(['hello', 'how', 'are', 'you'], word_2_idx))
    print(emb_mtx.shape)

    # Encode corpus dict, delete corpus dicts
    searchqa_enc_corpus_dict = encode_corpus_dict(searchqa_tok_corpus_dict, word_2_idx)
    del searchqa_tok_corpus_dict # to avoid memory errors
    quasar_enc_corpus_dict = encode_corpus_dict(quasar_tok_corpus_dict, word_2_idx)
    del quasar_tok_corpus_dict

    print('Encoded all corpus dictionaries.')
    return searchqa_enc_corpus_dict, quasar_enc_corpus_dict, emb_mtx, idx_2_word, word_2_idx

if __name__ == "__main__":
    searchqa_enc_corpus_dic, quasar_enc_corpus_dict, emb_mtx, idx_2_word, word_2_idx = main(process_glove=False)

    # Pickle:
    # Encoded SearchQA dict
    with open(os.path.join(OUTPUT_PATH_ENCODED, 'encoded_searchqa_dict.pkl'), 'wb') as fo:
        pickle.dump(searchqa_enc_corpus_dic, fo)
    # Encoded Quasar dict
    with open(os.path.join(OUTPUT_PATH_ENCODED, 'encoded_quasar_dict.pkl'), 'wb') as fo:
        pickle.dump(quasar_enc_corpus_dict, fo)
    # Embedding Matrix
    with open(os.path.join(OUTPUT_PATH_ENCODED, 'embedding.pkl'), 'wb') as fo:
        pickle.dump(emb_mtx, fo)
    # Index to Word dict
    with open(os.path.join(OUTPUT_PATH_ENCODED, 'idx_2_word_dict.pkl'), 'wb') as fo:
        pickle.dump(idx_2_word, fo)
    # Word to Index dict
    with open(os.path.join(OUTPUT_PATH_ENCODED, 'word_2_idx_dict.pkl'), 'wb') as fo:
        pickle.dump(word_2_idx, fo)

    print('Pickled and saved all files.')
'''
Notes on pytorch embedding layer    
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
'''