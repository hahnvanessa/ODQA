import question_answer_set as quas
import pickle
from collections import OrderedDict


quasar_train = 'encoded_quasar_dict.pkl'
quasar_dev = 'enc_quasar_dev_short.pkl'
quasar_test = 'enc_quasar_test_short.pkl'

searchqa_train = 'encoded_searchqa_dict.pkl'
searchqa_val = 'enc_searchqa_val.pkl'
searchqa_test = 'enc_searchqa_test.pkl'

def split_large_dataset(given_dict_path):
	'''Splits dataset that is too large to pickle into two halves and then processes it.
	Warning this is not guaranteed to keep order.
	But since both are part of training only this might not too big
	of an issue for now.''' 
	print(given_dict_path)
	filename_first_half = "qua_class_" + given_dict_path[:-4] + "_first_half.pkl"
	filename_second_half = "qua_class_" + given_dict_path[:-4] + "_second_half.pkl"
	with open(given_dict_path, 'rb') as fi:
		dict_data = pickle.load(fi)
		index_list = list(dict_data.items())
		first_half_dict = dict(index_list[len(dict_data)//2:])
		second_half_dict = dict(index_list[len(dict_data)//2:])
	
	first_half_set = quas.Question_Answer_Set(first_half_dict)
	second_half_set = quas.Question_Answer_Set(second_half_dict)
	
	f1 = open(filename_first_half, "wb")
	f2 = open(filename_second_half, "wb")
	
	print('Now storing {filename_first_half} and {filename_second_half} ...')
	pickle.dump(first_half_set,f1, protocol=4)
	f1.close()
	pickle.dump(second_half_set,f2,protocol=4)
	f2.close()
	return filename_first_half, filename_second_half
		

# Split and process largest dataset 
filename_first_half, filename_second_half = split_large_dataset(searchqa_train)
f1_1, f1_2 = split_large_dataset(filename_first_half)
f2_2, f2_2 = split_large_dataset(filename_second_half)

# Process all other datasets
for f in [quasar_train,quasar_dev,quasar_test,searchqa_val,searchqa_test]:
	with open(f, 'rb') as fi:
		n = "qua_class" + f 
		print('encoding', n)
		dict_data = pickle.load(fi)
		set_data = quas.Question_Answer_Set(dict_data)
		filehandler = open(n,"wb")
		pickle.dump(set_data,filehandler, protocol=4)
		filehandler.close()


