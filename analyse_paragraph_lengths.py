# Analyse the length distribution of the passages to determine optimal padding length
# Additionally, count the appearances of words
import pickle
import statistics
from collections import Counter

# Define the path to the pickled dictionaries
quasar_path = "quasar_train_short.pkl"
# todo: change searchqa path back to full train file
searchqa_path = "F:\\1QuestionAnswering\\preprocessed_files\\outputs\\searchqa_test.pkl"

def count_length_values(path, type='quasar') -> list:
    '''
    Function that counts the number of words per passage. Returns a list of these values. Includes punctuation.
    :param path:
    :param type:
    :return:
    '''
    # Length values
    length_values = []
    # quasar t train
    pickle_in = open(path,"rb")
    corpus_dict = pickle.load(pickle_in)
    # Keep track of the words that are added
    vocab_dict = Counter()
    length_values_quasar = []
    for question_id, qv in corpus_dict.items():
        if type == 'quasar':
            for _, context in qv['contexts']:
                # Note this also counts punctation but it should still return an approximate solution
                length_values.append(len(context.split()))
                vocab_dict.update(context.split())
        else:
            contexts = [c.split() for c in qv['contexts'] if c is not None]
            for c in contexts:
                length_values.append(len(c))
                vocab_dict.update(c)

    return length_values, vocab_dict

if __name__ == "__main__":
    # Quasar length passage
    length_values_quasar, _ = count_length_values(quasar_path, type='quasar')
    min_val_qst = min(length_values_quasar)
    max_val_qst = max(length_values_quasar)
    mean_val_qst = statistics.mean(length_values_quasar)
    median_val_qst = statistics.median(length_values_quasar)
    print("Quasar number of words and punctuation per passage: Min length is <{}> max length is <{}> \n\r mean length is <{}> and median length is <{}>.".format(min_val_qst, max_val_qst, mean_val_qst, median_val_qst))

    # Searchqa length passage
    length_values_searchqa, _ = count_length_values(searchqa_path, type='searchqa')
    min_val_sqa = min(length_values_searchqa)
    max_val_sqa = max(length_values_searchqa)
    mean_val_sqa = statistics.mean(length_values_searchqa)
    median_val_sqa = statistics.median(length_values_searchqa)

    print("Searchqa length of passage (includes punctuation): Min length is <{}> max length is <{}> \n\r mean length is <{}> and median length is <{}>.".format(min_val_sqa, max_val_sqa, mean_val_sqa, median_val_sqa))
