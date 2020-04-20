# os and sys
import os
import sys
# numpy
import numpy as np
import bcolz
import pickle
from collections import Counter
import spacy
import torch

# Disable spacy components to speed up tokenization
nlp = spacy.load('en_core_web_sm',
                 disable=['tagger', 'parser', 'ner', 'print_info'])
# todo: multiprocessing
# Handle the file paths
from pathlib import Path
# Typing
from typing import Tuple

dirpath = os.getcwd()
GLOVE_FILE = Path("/".join([dirpath, 'glove.840B.300d.txt']))
GLOVE_PATH = Path(dirpath)

# Filepaths to files that will be used to create the embedding matrix
DATASET_PATH_SEARCHQA = Path("/".join([dirpath, 'outputs', 'searchqa_train.pkl']))  # pickled
DATASET_PATH_QUASAR = Path("/".join([dirpath, 'outputs', 'quasar_train_short.pkl']))  # pickled
# Output Pathes
OUTPUT_PATH_ENCODED = Path("/".join([dirpath, 'outputs']))

# Parameters
MAX_SEQUENCE_LENGTH = 100  # should approximately be the mean sentence length+ 0.5std, searchqa has highest seq. length with appr. 54
VOCAB_SIZE = 100000  # todo: decide on vocab size, not specified by paper, good value might be 20000
EMBEDDING_DIM = 300  # 300 dimensions as specified by paper
PAD_IDENTIFIER = '<PAD>'
UNKNOWN_IDENTIFIER = '<UNK>'


def build_glove_dict() -> dict:
    '''
    Builds a dictionary based on a glove file that maps words to embedding vectors.
    :return:
    '''
    print("Building and pickling glove vectors.")
    # List of words that appear in the Glove file
    words = []
    # Index
    idx = 0
    # Word to index mapping
    word2idx = {}
    # This will be the file where we store our vectors, words and indexes are stored in separate
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{GLOVE_PATH}/840B.300d.dat', mode='w')

    with open(GLOVE_FILE, 'r', errors='ignore', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            word = " ".join(line[:-EMBEDDING_DIM])
            words.append(word)
            word2idx[word] = idx
            vect = np.array(line[-EMBEDDING_DIM:]).astype(np.float)
            vectors.append(vect)
            if idx % 10000 == 0 and idx > 0:
                print("Read in {} glove vectors of about 2,200,000 in total.".format(idx))
            idx += 1  # comment out for testing

    vectors = bcolz.carray(vectors[1:].reshape((-1, EMBEDDING_DIM)), rootdir=f'{GLOVE_PATH}\\840B.300d.dat', mode='w')
    # Write vectors to disk
    vectors.flush()
    # Save word list to file
    pickle.dump(words, open(f'{GLOVE_PATH}/840B.300d_words.pkl', 'wb'))
    # save index list to file
    pickle.dump(word2idx, open(f'{GLOVE_PATH}/840B.300d_idx.pkl', 'wb'))

    # Build vocab that maps index to word
    glove = {w: vectors[word2idx[w]] for w in words}
    pickle.dump(glove, open(f'{GLOVE_PATH}/glove_dict.pkl', 'wb'))
    # Return the glove dictionary
    return glove


def load_pickled_glove():
    return pickle.load(open(f'{GLOVE_PATH}/glove_dict.pkl', 'rb'))


def tokenize_context(context) -> list:
    '''
    Tokenizes the given context by using spacy tokenizer.
    :param context:
    :return:
    '''
    return [token.text for token in nlp(context)]


def tokenize_set(DATASET_PATH, type='quasar') -> Tuple[dict, int]:
    print("Applying tokenization to set: {}".format(type))
    # todo: check if it is efficient and indexwise ok to store all contexts in a simple list, alternative: array
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
        # Tokenize question
        corpus_dict[question_id]['tokenized_question'] = tokenize_context(qv['question'])
        token_count.update(corpus_dict[question_id]['tokenized_question'])
        # Tokenize answer
        corpus_dict[question_id]['tokenized_answer'] = tokenize_context(qv['answer'])
        token_count.update(corpus_dict[question_id]['tokenized_answer'])

        i += 1
        if i % 1000 == 0:
            print('Tokenized {} of {} questions in total for set <{}>'.format(i, len(corpus_dict), type))
    return corpus_dict, token_count


def make_emedding_matrix(glove_dict, target_vocab):
    '''
    Returns a matrix of size VOCAB_SIZE*EMBEDDING_DIM.
    Target vocabulary are the top VOCAB_SIZE most occuring words
    in a set.
    :param glove_dict:
    :param target_vocab:
    :return:
    '''
    print("Building embedding matrix.")
    matrix_len = len(target_vocab)
    # Weight matrix has vocabulary plus the entries for padding and unknown
    weights_matrix = np.zeros((matrix_len + 2, EMBEDDING_DIM))
    idx_2_word = {}
    word_2_idx = {}
    words_found = 0

    # Add padding symbol
    idx_2_word[0] = PAD_IDENTIFIER
    word_2_idx[PAD_IDENTIFIER] = 0
    pad_vector = np.zeros(shape=(EMBEDDING_DIM,))
    weights_matrix[0] = pad_vector
    # Add unknown word (word that does not appear in the top vocabulary)
    idx_2_word[1] = UNKNOWN_IDENTIFIER
    word_2_idx[UNKNOWN_IDENTIFIER] = 1
    unknown_vector = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
    weights_matrix[1] = unknown_vector

    for i, word in enumerate(target_vocab, start=2):
        try:
            weights_matrix[i] = glove_dict[word]
            words_found += 1
        # If the glove dictionary does not not contain the word, add random vector
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))  # scale refers to standard deviation
        idx_2_word[i] = word
        word_2_idx[word] = i
    print('{} of the {} words in the quasar/searchqa set were found in the glove set'.format(words_found, matrix_len))
    return weights_matrix, idx_2_word, word_2_idx

# todo: change name of function to encode_pad
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
    tokenized_context.extend([PAD_IDENTIFIER] * MAX_SEQUENCE_LENGTH)  # always add so min size is always MAX_SEQUENCE_LENGTH + 1
    tokenized_context = tokenized_context[:MAX_SEQUENCE_LENGTH]
    # Index words, identify unkown ones
    for t in tokenized_context:
        # If word in top vocabulary or is a padding token
        if t in word_2_idx:
            encoded_context.append(word_2_idx[t])
        else:
            encoded_context.append(word_2_idx[UNKNOWN_IDENTIFIER])
    # todo: Change this back to a different format
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
            encoded_context = encode_pad_context(tokenized_context, word_2_idx)
            corpus_dict[question_id]['encoded_contexts'].append(encoded_context)
        # Encode question
        encoded_question = encode_pad_context(corpus_dict[question_id]['tokenized_question'], word_2_idx)
        corpus_dict[question_id]['encoded_question'] = encoded_question
        del corpus_dict[question_id]['tokenized_question']
        # Encode answer
        encoded_answer = encode_pad_context(corpus_dict[question_id]['tokenized_answer'], word_2_idx)
        corpus_dict[question_id]['encoded_answer'] = encoded_answer
        del corpus_dict[question_id]['tokenized_answer']
        i += 1
        if i % 1000 == 0:
            print('Encoded {} of {} questions in total'.format(i, len(corpus_dict)))
        # Delete tokenized entries to save memory
        del corpus_dict[question_id]['tokenized_contexts']

    return corpus_dict

def encode_untokenized_file(DATASET_PATH, filename, word_2_idx, type='quasar'):
    '''
    Tokenizes and encodes a single file. This can be used to encode val, test or toy files.
    Also saves to disk.
    :param filepath:
    :param type:
    :param word_2_idx:
    :return:
    '''
    assert type in ['quasar', 'searchqa'], 'Wrong type specified. Allowed types are "quasar" and "searchqa'

    # Tokenize
    tok_corpus_dict, _ = tokenize_set(DATASET_PATH, type=type)
    # Encode
    enc_corpus_dict = encode_corpus_dict(tok_corpus_dict, word_2_idx)
    # Save to disk
    with open(os.path.join(OUTPUT_PATH_ENCODED, filename), 'wb') as fo:
        pickle.dump(enc_corpus_dict, fo)

    print('wrote file to disk', filename)

    return enc_corpus_dict


def load_matrix_and_mapping_dictionaries():
    '''
    Loads the embedding matrix, the index2word dictionary and the word2index dictionary from disk.
    :return:
    '''
    emb_mtx = pickle.load(open(f'{OUTPUT_PATH_ENCODED}/embedding_matrix.pkl', 'rb'))
    idx_2_word = pickle.load(open(f'{OUTPUT_PATH_ENCODED}/idx_2_word_dict.pkl', 'rb'))
    word_2_idx = pickle.load(open(f'{OUTPUT_PATH_ENCODED}/word_2_idx_dict.pkl', 'rb'))
    return emb_mtx, idx_2_word, word_2_idx


def main(process_glove=False, tokenize=False, encode=False):
    '''
    Performs processing of the glove file, tokenization of the (large) training corpora and
    building of the embedding matrix and the word/indx mapping dictionaries. If an attribute
    is set to False it will load preproccessed files instead.
    Returns the encoded corpora, the embedding matrix and the mappings.
    :param process_glove:
    :param tokenize:
    :param encode:
    :return:
    '''
    # Specify whether the original Glove file should be processed or a
    # already pickled dictionary version of Glove should be loaded
    if process_glove:
        glove = build_glove_dict()
    else:
        glove = load_pickled_glove()

    # Tokenization and Embedding Matrix Creation
    if tokenize:
        # 1. Retrieve most used vocabulary words from searchqa and quasar
        # 2. Tokenize contexts but do not apply padding yet, tokenized contexts stored in corpus_dict
        searchqa_tok_corpus_dict, searchqa_token_count = tokenize_set(DATASET_PATH_SEARCHQA, type='searchqa')
        quasar_tok_corpus_dict, quasar_token_count = tokenize_set(DATASET_PATH_QUASAR, type='quasar')
        # 3. Combine the top vocabularies from both sets
        total_token_count = searchqa_token_count + quasar_token_count
        top_vocabulary = [x[0] for x in total_token_count.most_common(VOCAB_SIZE)]

        # Create an embedding matrix
        emb_mtx, idx_2_word, word_2_idx = make_emedding_matrix(glove_dict=glove, target_vocab=top_vocabulary)

        # Embedding Matrix
        with open(os.path.join(OUTPUT_PATH_ENCODED, 'embedding_matrix.pkl'), 'wb') as fo:
            pickle.dump(emb_mtx, fo)
        # Index to Word dict
        with open(os.path.join(OUTPUT_PATH_ENCODED, 'idx_2_word_dict.pkl'), 'wb') as fo:
            pickle.dump(idx_2_word, fo)
        # Word to Index dict
        with open(os.path.join(OUTPUT_PATH_ENCODED, 'word_2_idx_dict.pkl'), 'wb') as fo:
            pickle.dump(word_2_idx, fo)
        # Save the tokenized dicts
        with open(os.path.join(OUTPUT_PATH_ENCODED, 'tokenized_searchqa_dict.pkl'), 'wb') as fo:
           pickle.dump(searchqa_tok_corpus_dict, fo)
        # Encoded Quasar dict
        with open(os.path.join(OUTPUT_PATH_ENCODED, 'tokenized_quasar_dict.pkl'), 'wb') as fo:
           pickle.dump(quasar_tok_corpus_dict, fo)
        print('Tokenized corpus dictionaries, created embedding matrix and mapping dictionaries, pickled them.')

    else:
        emb_mtx, idx_2_word, word_2_idx =  load_matrix_and_mapping_dictionaries()
        searchqa_tok_corpus_dict = pickle.load(open(f'{OUTPUT_PATH_ENCODED}/tokenized_searchqa_dict.pkl', 'rb'))
        quasar_tok_corpus_dict = pickle.load(open(f'{OUTPUT_PATH_ENCODED}/tokenized_quasar_dict.pkl', 'rb'))
        print('Loaded tokenized dictionaries')

    # Test encoding
    print(encode_pad_context(['hello', 'how', 'are', 'you'], word_2_idx))
    print(emb_mtx.shape)

    # Encode corpus dict, delete corpus dicts
    if encode:
        searchqa_enc_corpus_dict = encode_corpus_dict(searchqa_tok_corpus_dict, word_2_idx)
        del searchqa_tok_corpus_dict  # to avoid memory errors
        quasar_enc_corpus_dict = encode_corpus_dict(quasar_tok_corpus_dict, word_2_idx)
        del quasar_tok_corpus_dict

        # Save Encoded SearchQA dict
        with open(os.path.join(OUTPUT_PATH_ENCODED, 'encoded_searchqa_dict.pkl'), 'wb') as fo:
            pickle.dump(searchqa_enc_corpus_dict, fo)
        # Save Encoded Quasar dict
        with open(os.path.join(OUTPUT_PATH_ENCODED, 'encoded_quasar_dict.pkl'), 'wb') as fo:
            pickle.dump(quasar_enc_corpus_dict, fo)

        print('Encoded all corpus dictionaries.')
    else:
        searchqa_enc_corpus_dict = pickle.load(open(f'{OUTPUT_PATH_ENCODED}/encoded_searchqa_dict.pkl', 'rb'))
        quasar_enc_corpus_dict = pickle.load(open(f'{OUTPUT_PATH_ENCODED}/encoded_quasar_dict.pkl', 'rb'))
        print('Loaded encoded corpus dictionaries')

    return searchqa_enc_corpus_dict, quasar_enc_corpus_dict, emb_mtx, idx_2_word, word_2_idx


if __name__ == "__main__":
    # Run main to create encoded training files, embedding matrix, word/index mappings
    searchqa_enc_corpus_dic, quasar_enc_corpus_dict, emb_mtx, idx_2_word, word_2_idx = main(process_glove=False, tokenize=True, encode=True)
    # Encode all other files that need to be encoded

    # Input paths
    SEARCHQA_VAL = Path("/".join([dirpath, 'outputs', 'searchqa_val.pkl']))
    SEARCHQA_TEST = Path("/".join([dirpath, 'outputs', 'searchqa_test.pkl']))
    QUASAR_DEV = Path("/".join([dirpath, 'outputs', 'quasar_dev_short.pkl']))
    QUASAR_TEST = Path("/".join([dirpath, 'outputs', 'quasar_test_short.pkl']))

    # Output filenames
    ENC_SEARCHQA_VAL = 'enc_searchqa_val.pkl'
    ENC_SEARCHQA_TEST = 'enc_searchqa_test.pkl'
    ENC_QUASAR_DEV =  'enc_quasar_dev_short.pkl'
    ENC_QUASAR_TEST =  'enc_quasar_test_short.pkl'


    enc_searchqa_val = encode_untokenized_file(SEARCHQA_VAL, ENC_SEARCHQA_VAL, word_2_idx, type='searchqa')
    enc_searchqa_test = encode_untokenized_file(SEARCHQA_TEST, ENC_SEARCHQA_TEST, word_2_idx, type='searchqa')

    enc_quasar_dev = encode_untokenized_file(QUASAR_DEV, ENC_QUASAR_DEV, word_2_idx, type='quasar')
    enc_quasar_test = encode_untokenized_file(QUASAR_TEST, ENC_QUASAR_TEST, word_2_idx, type='quasar')
