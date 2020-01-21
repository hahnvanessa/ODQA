# Analyse the length distribution of the passages to determine optimal padding length
import pickle
import statistics

# Define the path to the pickled dictionaries
quasar_path = "quasar_train_short.pkl"
searchqa_path = "F:\\1QuestionAnswering\\preprocessed_files\\outputs\\searchqa_train.pkl"

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
    length_values_quasar = []
    for question_id, qv in corpus_dict.items():


        if type == 'quasar':
            for _, context in qv['contexts']:
                # Note this also counts punctation but it should still return an approximate solution
                length_values.append(len(context.split()))
        else:
            # todo: searchq seems to contain empty answer sets. Check this.
            length_values.extend([len(c.split()) for c in qv['contexts'] if c is not None])
    return length_values

# Quasar length passage
length_values_quasar = count_length_values(quasar_path, type='quasar')
min_val_qst = min(length_values_quasar)
max_val_qst = max(length_values_quasar)
mean_val_qst = statistics.mean(length_values_quasar)
median_val_qst = statistics.median(length_values_quasar)
print("Quasar length of passage (includes punctuation): Min length is <{}> max length is <{}> \n\r mean length is <{}> and median length is <{}>.".format(min_val_qst, max_val_qst, mean_val_qst, median_val_qst))

# Searchqa length passage
length_values_searchqa = count_length_values(searchqa_path, type='searchqa')
min_val_sqa = min(length_values_searchqa)
max_val_sqa = max(length_values_searchqa)
mean_val_sqa = statistics.mean(length_values_searchqa)
median_val_sqa = statistics.median(length_values_searchqa)

print("Searchqa length of passage (includes punctuation): Min length is <{}> max length is <{}> \n\r mean length is <{}> and median length is <{}>.".format(min_val_sqa, max_val_sqa, mean_val_sqa, median_val_sqa))
