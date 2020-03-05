from argparse import ArgumentParser
from BILSTM import BiLSTM, attention
import os
import pickle
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import question_answer_set
from torch.utils.data import Dataset, DataLoader
import question_answer_set as qas
import candidate_scoring

MAX_SEQUENCE_LENGTH = 100

def get_file_paths(data_dir):
    # Get paths for all files in the given directory
    file_names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(data_dir):
        for file in f:
            if '.pkl' in file:
                file_names.append(os.path.join(r, file))
    return file_names

def reward(c, a, c_len, a_len):
    '''
    Returns the reward for a candidate prediction.
    todo: Make this capabele of using batches
    :param candidate:
    :param answer:
    :return:
    '''

    def f1_score(c, a, c_len, a_len, epsilon=1e-7):
        # source for reference: https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
        y_true = torch.ones(c.shape, dtype=torch.bool).long()
        y_pred = torch.eq(c, a).long()
        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)


        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        return f1

    # Trim padding to candidate len
    assert c_len >= a_len
    c = c[0, :c_len]
    a = a[0, :c_len]

    if torch.all(torch.eq(c, a)):
        return 2
    elif len(torch.eq(c, a).nonzero()) > 0:
        return f1_score(c, a, c_len, a_len)
    else:
        return -1

def batch_training(dataset, embedding_matrix, batch_size=6, num_epochs=10):
    '''
    Performs minibatch training.
    :param dataset:
    :param embedding_matrix:
    :param batch_size:
    :param num_epochs:
    :return:
    '''
    # Load Dataset with the dataloader
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Store representations
    qp_representations = {}
    int_representations = {}
    candidate_scores = {}

    # Initialize BiLSTMs
    qp_bilstm = BiLSTM(embedding_matrix, embedding_dim=300, hidden_dim=100,
                batch_size=batch_size)
    interaction_bilstm = BiLSTM(embedding_matrix, embedding_dim=400, hidden_dim=100,
                batch_size=batch_size) #check this batch size
    G_bilstm = nn.LSTM(input_size=400, hidden_size=100, bidirectional=True)


    for epoch in range(num_epochs):

        for batch_number, data in enumerate(train_loader):
            questions, contexts, answers, q_len, c_len, a_len, q_id = data
            # Pack (reduce the sentences to their original form without padding)
            packed_q = torch.nn.utils.rnn.pack_padded_sequence(questions, q_len, batch_first=True, enforce_sorted=False)
            packed_c = torch.nn.utils.rnn.pack_padded_sequence(contexts, c_len, batch_first=True, enforce_sorted=False)
            packed_a = torch.nn.utils.rnn.pack_padded_sequence(answers, a_len, batch_first=True, enforce_sorted=False)
            # Question and Passage Representation
            q_representation = qp_bilstm.forward(questions)
            c_representation = qp_bilstm.forward(contexts)
            # Question and Passage Interaction
            HP_attention = attention(q_representation, c_representation)
            G_input = torch.cat((c_representation, HP_attention), 2)
            G_ps, _ = G_bilstm.forward(G_input)

            scores = []  # store all candidate scores for each context for the current question
            for G_p in G_ps:
                print('gp', G_p.shape)
                input()
                # create a new Candidate Scorer for each context
                C_scores = candidate_scoring.Candidate_Scorer(G_p).candidate_probabilities()  # candidate scores for current context
                print('cscores', C_scores.shape)
                input()
                scores.append(C_scores)
            # if we create only one candidate scorer instance before (e.g. one for each question or one for all questions), we need to change the G_p argument
            print('scores', scores)

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
