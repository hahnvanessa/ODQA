import question_answer_set as quas
import pickle
from collections import OrderedDict
from os import walk
from argparse import ArgumentParser
import os
import sys
from pathlib import Path

'''
Turns all dictionary files (train/dev/test) into QAS files.
'''

quasar_train = 'encoded_quasar_dict.pkl'
quasar_dev = 'enc_quasar_dev_short.pkl'
quasar_test = 'enc_quasar_test_short.pkl'

searchqa_train = 'encoded_searchqa_dict.pkl'
searchqa_val = 'enc_searchqa_val.pkl'
searchqa_test = 'enc_searchqa_test.pkl'

def main(dirpath):
    for (dirpath, dirnames, filenames) in walk(dirpath):
        for f in filenames:
            print(f)
            if 'enc' in f:
                with open(Path("/".join([dirpath, f])), 'rb') as fi:
                    n = "qua_class" + f[3:] 
                    output_path = Path("/".join([dirpath, n]))
                    print('encoding', n)
                    dict_data = pickle.load(fi)
                    set_data = quas.Question_Answer_Set(dict_data)
                    filehandler = open(output_path,"wb")
                    pickle.dump(set_data,filehandler, protocol=4)
                    filehandler.close()
                    
if __name__ == "__main__":
    parser = ArgumentParser(description='Turns all encoded dict files into question answer set objects')
    parser.add_argument('--out', default='/local/fgoessl/outputs', type=str, help='Path to output directory')
    # Parse given arguments
    args = parser.parse_args()
    print(args.out)
    
    main(dirpath=args.out)
