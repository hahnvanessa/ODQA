from argparse import ArgumentParser
from utils.BILSTM import BiLSTM, attention, max_pooling
import os
import pickle
import numpy as np
from tqdm import tqdm
#torch
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
from torch.optim import lr_scheduler
from model.model import ODQA
from torch import nn, optim
# utils
import utils.question_answer_set as question_answer_set
from utils.loss import Loss_Function
import utils.rename_unpickler as ru
from utils.pretraining import remove_data, pretrain_candidate_scoring, pretrain_answer_selection
# Init wandb
import wandb
wandb.init(project="ODQA")

MAX_SEQUENCE_LENGTH = 100
K = 2 # Number of extracted candidates per passage

# todo: fix the paths here
with open('/local/fgoessl/outputs/outputs_v4/idx_2_word_dict.pkl', 'rb') as f:
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
   
def freeze_candidate_extraction(model):
    ''' Freezes the parameters in the candidate extraction part of the model'''
    for p in model.qp_bilstm.parameters():
        p.requires_grad = False
    for p in model.G_bilstm.parameters():
        p.requires_grad = False
    for p in model.candidate_scorer.wb.parameters():
        p.requires_grad = False
    for p in model.candidate_scorer.we.parameters():
        p.requires_grad = False

def get_distance(passages, candidates):
    ''' Distance feature for advanced passage representation between each position on passage and entire candidate'''
    passage_distances = []
    length = candidates.shape[0]
    for i in range(length):
        position_distances = []
        for p in range(passages.shape[1]):
            position_distances.append(torch.dist(passages[i,p,:], candidates[i,:,:]))
        position_distances = torch.stack(position_distances, dim=0)
        passage_distances.append(position_distances.view(1,passages.shape[1]))
    return torch.squeeze(torch.stack(passage_distances, dim=0))


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
def batch_training(dataset, embedding_matrix, pretrained_parameters_filepath=None, batch_size=100, num_epochs=10):
    '''
    Performs minibatch training. One datapoint is a question-context-answer pair.
    :param dataset:
    :param embedding_matrix:
    :param batch_size:
    :param num_epochs:
    :return:
    '''
    # Offer a sacrifice to the Cuda-God so that it may reward us with high accuracy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_matrix = torch.Tensor(embedding_matrix).to(device)

    # Load Dataset with the dataloader
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8) #num_workers = 4 * num_gpu, but divide by half cuz sharing is caring

    # Initialize model
    if pretrained_parameters_filepath == None:
        model = ODQA(k=K, max_sequence_length=MAX_SEQUENCE_LENGTH, batch_size=batch_size, embedding_matrix=embedding_matrix, device=device).to(device)	
    else:
        model = ODQA(k=K, max_sequence_length=MAX_SEQUENCE_LENGTH, batch_size=batch_size, embedding_matrix=embedding_matrix, device=device).to(device)
        model.load_parameters(filepath=pretrained_parameters_filepath)
        model.reset_batch_size(batch_size)


    # torch settings
    #todo: check wheter ALL our parameters are in there e.g. candiate_representation
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    #todo: set these to proper values
    optimizer = optim.RMSprop(parameters, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.00001, centered=False)
    criterion = nn.CrossEntropyLoss() #https://stackoverflow.com/questions/49390842/cross-entropy-in-pytorch

    for epoch in range(num_epochs):
        for batch_number, data in enumerate(train_loader):
            print(f'epoch number {epoch} batch number {batch_number}.')
            predicted_answer, question, ground_truth_answer = model.forward(data)
            predicted_answer_as_strings = candidate_to_string(predicted_answer)
            ground_truth_answer_as_strings = candidate_to_string(ground_truth_answer)
            question_as_strings = candidate_to_string(question)
            print(question_as_strings, predicted_answer_as_strings, ground_truth_answer_as_strings)

            '''
            optimizer.zero_grad()
            batch_loss = Loss_Function.loss(predicted_answer, ground_truth_answer)
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            '''
    # Save optimized parameters
    model.store_parameters('test_file_parameters.pth')



def pretraining(dataset, embedding_matrix, pretrained_parameters_filepath, num_epochs, batch_size):
    '''
    Performs minibatch training. One datapoint is a question-context-answer pair.
    :param dataset:
    :param embedding_matrix:
    :param batch_size:
    :param num_epochs:
    :return:
    '''
    # Offer a sacrifice to the Cuda-God so that it may reward us with high accuracy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_matrix = torch.Tensor(embedding_matrix)



    # Load Dataset with the dataloader
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8) #num_workers = 4 * num_gpu, but divide by half cuz sharing is caring

    # Initialize model
    model = ODQA(k=K, max_sequence_length=MAX_SEQUENCE_LENGTH, batch_size=batch_size, embedding_matrix=embedding_matrix, device=device).to(device)	
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    #todo: set these to proper values
    optimizer = optim.RMSprop(parameters, lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1, last_epoch=-1)
    criterion = nn.CrossEntropyLoss() #https://stackoverflow.com/questions/49390842/cross-entropy-in-pytorch https://stackoverflow.com/questions/53936136/pytorch-inputs-for-nn-crossentropyloss
    criterion.requires_grad = True
    loss = 0
    step = 0
    gd_batch = 0

    for epoch in range(num_epochs):
        for batch_number, data in enumerate(tqdm(train_loader)):          
            #print(f'pretraining poch number {epoch}/{num_epochs} batch number {batch_number}.')
            data = remove_data(data, remove_passages='no_ground_truth')
            if len(data[0]) != 0:
                 gd_batch += 1
                 k_max_list, gt_span_idxs = pretrain_candidate_scoring(model, data, MAX_SEQUENCE_LENGTH)
                 batch_loss = criterion(k_max_list,gt_span_idxs)
                 #print('batch loss....: ', batch_loss)	
                 optimizer.zero_grad()
                 loss += batch_loss.item()
                 batch_loss.backward()
                

                 optimizer.step()
                 scheduler.step(batch_loss)
                
                 if gd_batch != 0 and gd_batch % 100 == 0:
                     av_loss = loss / 100
                     #print(f'total loss per step {loss}')
                     loss = 0
                     wandb.log({'pretraining loss (extraction)': av_loss, 
                                'lr': scheduler.get_lr()}, step=step)
                     step += 1
        scheduler.step()
    model.store_parameters('test_file_parameters.pth')


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
    print('embedding matrix loaded')

    # Retrieve the filepaths of all encoded corpora
    file_paths = get_file_paths(encoded_corpora)

    qp_representations = {}
    int_representations = {}

    # Testing
    testfile = '/local/fgoessl/outputs/outputs_v4/QUA_Class_files/qua_classenc_quasar_dev_short.pkl' #qua_classenc_quasar_dev_short.pkl' 
    with open(testfile, 'rb') as f:
        print('Loading', f)
        dataset = ru.renamed_load(f)

        # Minibatch training
        pretraining(dataset, embedding_matrix, pretrained_parameters_filepath=None, batch_size=100, num_epochs=args.num_epochs)

  
    '''
    for file in file_paths:
        with open(os.path.join(file), 'rb') as f:
            print('Loading... ', f)
            dataset = ru.renamed_load(f)
            print(f)
            # Minibatch training
            batch_training(dataset, embedding_matrix, pretrained_parameters_filepath=None, batch_size=100, num_epochs=10)
    '''
if __name__ == '__main__':
   
    parser = ArgumentParser(
        description='Main ODQA script')
    parser.add_argument(
        '--lr', default=0.00001, type=float, help='Learning rate value')
    parser.add_argument(
        '--num_epochs', default=1, type=int, help='The number of training epochs')

    # Parse given arguments
    args = parser.parse_args()

    # Call main()
    #main(embedding_matrix=args.embeddings, encoded_corpora=args.data)
    main(embedding_matrix='/local/fgoessl/outputs/outputs_v4/embedding_matrix.pkl', encoded_corpora='/local/fgoessl/outputs/outputs_v4/QUA_Class_files')
