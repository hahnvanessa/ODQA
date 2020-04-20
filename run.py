from argparse import ArgumentParser
from utils.BILSTM import BiLSTM, attention, max_pooling
import os
import pickle
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import utils.question_answer_set as qas
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
from model.model import ODQA

MAX_SEQUENCE_LENGTH = 100
K = 2 # Number of extracted candidates per passage

# todo: fix the paths here
with open('outputs_numpy_encoding_v2//idx_2_word_dict.pkl', 'rb') as f:
    idx_2_word_dic = pickle.load(f)

def candidate_to_string(candidate, idx_2_word_dic=idx_2_word_dic):
    '''
    Turns a tensor of indices into a string. Basically gives us back. Can be used
    to turn our candidates back into sentences.
    :param candidate:
    :param idx_2_word_dic:
    :return:
    '''
    '''
    Example:
    values, indices = torch.max(p_C, 0) #0 indicates the dimension along which you want to find the max
    # todo: assert that only one value is returned
    print(values, indices)
    print(candidate_to_string(encoded_candidates[indices]))
    '''
    return [idx_2_word_dic[i] for i in candidate.tolist() if i != 0]
   


def get_distance(passages, candidates):
    passage_distances = []
    length = candidates.shape[0]
    for i in range(length):
        position_distances = []
        for p in range(passages.shape[1]):
            position_distances.append(torch.dist(passages[i,p,:], candidates[i,:,:]))
        position_distances = torch.stack(position_distances, dim=0)
        passage_distances.append(position_distances.view(1,passages.shape[1]))
    return torch.squeeze(torch.stack(passage_distances, dim=0))

def store_model(model, filepath):
    # todo
    pass


def load_model(filepath):
    # todo
    pass

def get_file_paths(data_dir):
    # Get paths for all files in the given directory
    file_names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(data_dir):
        for file in f:
            if '.pkl' in file:
                file_names.append(os.path.join(r, file))
    return file_names


# todo: batch size must be varied manually depending on whether we use searchqa or quasar
def batch_training(dataset, embedding_matrix, pretrained_model=None, batch_size=100, num_epochs=10):
    '''
    Performs minibatch training. One datapoint is a question-context-answer pair.
    :param dataset:
    :param embedding_matrix:
    :param batch_size:
    :param num_epochs:
    :return:
    '''
    # Load Dataset with the dataloader
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Offer a sacrifice to the Cuda-God so that it may reward us with high accuracy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #todo: check if ressources of gpu are really used

    # Initialize model
    if pretrained_model == None:
        model = ODQA(k=K, max_sequence_length=MAX_SEQUENCE_LENGTH, batch_size=batch_size, embedding_matrix=embedding_matrix).to(device)
    else:
        model = pretrained_model
        model.reset_batch_size(batch_size=batch_size)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    for epoch in range(num_epochs):
        for batch_number, data in enumerate(train_loader):
            print(f'epoch number {epoch} batch number {batch_number}.')
            predicted_answer, question, ground_truth_answer = model.forward(data)
            predicted_answer_as_strings = candidate_to_string(predicted_answer)
            ground_truth_answer_as_strings = candidate_to_string(ground_truth_answer)
            question_as_strings = candidate_to_string(question)
            #print(question_as_strings, predicted_answer_as_strings, ground_truth_answer_as_strings)



'''
def test(model, dataset, batch_size):
    
    Test on dev set.
    :param model:
    :return:
    
    # Load dataset
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Disable gradient as we do not conduct backpropagation
    with torch.set_grad_enabled(False):
        for batch_number, data in enumerate(train_loader):
            model.forward(data)
            batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
            loss += batch_loss.item()

            # (batch, c_len, c_len)
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1,
                                                                                                      -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for i in range(batch_size):
                id = batch.id[i]
                answer = batch.c_word[0][i][s_idx[i]:e_idx[i] + 1]
                answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
                answers[id] = answer

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup_params.get(name))

    with open(args.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)

    results = evaluate.main(args)
    return loss, results['exact_match'], results['f1']
'''

def main(embedding_matrix, encoded_corpora):
    '''
    Iterates through all given corpus files and forwards the encoded contexts and questions
    through the BILSTMs.
    :param embedding_matrix:
    :param encoded_corpora:
    :return:
    '''

    embedding_matrix = pickle.load(open(embedding_matrix, 'rb'))

    # Retrieve the filepaths of all encoded corpora
    file_paths = get_file_paths(encoded_corpora)

    qp_representations = {}
    int_representations = {}

    for file in file_paths:
        with open(os.path.join(file), 'rb') as f:
            content = pickle.load(f)

            # Minibatch training
            dataset = qas.Question_Answer_Set(content)
            batch_training(dataset, embedding_matrix, batch_size=100, num_epochs=10)

if __name__ == '__main__':
    '''
    parser = ArgumentParser(
        description='Main ODQA script')
    parser.add_argument(
        'embeddings', help='Path to the pkl file')
    parser.add_argument(
        'data', help='Path to the folder with the pkl files')

    # Parse given arguments
    args = parser.parse_args()
    '''
    # Call main()
    #main(embedding_matrix=args.embeddings, encoded_corpora=args.data)
    main(embedding_matrix='embedding_matrix.pkl', encoded_corpora='outputs_numpy_encoding_v2')
