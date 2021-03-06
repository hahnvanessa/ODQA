# Open Domain Question Answering System with Joint Training using Reinforcement Learning

Reimplementation of https://www.aclweb.org/anthology/P18-1159/ for the course Advances in Question Answering, WS 19/20, Saarland University by Ekaterina Saveleva, Fabian Goessl and Vanessa Hahn

## Installation Requirements 
```bash
pip install -r requirements.txt
```

## Data preprocessing
### Data conversion
Use a variation of this command for every file to convert and save it as a pickled dictionary object:
```python
python3 preprocessing.py -t "searchqa" -f /local/user/output/searchQA  -s "test"
```
### Encoding
For encoding please add a glove file and all converted files in output, then run:
```python
python3 embeddings.py --out /local/user/output 
```

### Preprocessing and Dataset objects
To do the final preprocessing of all the encoded files call:
```python
python dict_to_qas.py --out /local/user/output 
```
## Usage 
Collect training and test files in two separate folders. Example of usage: 

```python
python run.py --lr 0.0001 --num_epochs 2 --emb /local/user/embedding_matrix.pkl --id2v /local/user/idx_2_word_dict.pkl --input_train /local/user/train_files_folder/ --input_test /local/user/test_files_folder/
```

## Data
SearchQA: https://github.com/nyu-dl/dl4ir-searchqA

Quasar-T: https://github.com/bdhingra/quasar
