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

def batch_training(dataset, embedding_matrix, batch_size=100, num_epochs=10):
    '''
    Performs minibatch training. One datapoint is a question-context-answer pair.
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
    G_bilstm = nn.LSTM(input_size=400, hidden_size=100, bidirectional=True)


    for epoch in range(num_epochs):

        for batch_number, data in enumerate(train_loader):
            questions, contexts, answers, q_len, c_len, a_len, q_id, common_word_encodings = data
            # Question and Passage Representation
            q_representation = qp_bilstm.forward(questions, sentence_lengths=q_len)
            c_representation = qp_bilstm.forward(contexts, sentence_lengths=c_len)
            # Question and Passage Interaction
            HP_attention = attention(q_representation, c_representation)
            G_input = torch.cat((c_representation, HP_attention), 2)
            G_ps, _ = G_bilstm.forward(G_input)

            scores = []  # store all candidate scores for each context for the current question
            for G_p in G_ps:
                # create a new Candidate Scorer for each context
                C_scores = candidate_scoring.Candidate_Scorer(G_p).candidate_probabilities()  # candidate scores for current context
                scores.append(C_scores)
            # if we create only one candidate scorer instance before (e.g. one for each question or one for all questions), we need to change the G_p argument
            print('scores', scores)

            # Question Representation


            # Passage Representation
            S_q = qp_bilstm.forward(questions, sentence_lengths=q_len)
            mxp = nn.MaxPool2d((100, 1), stride=1) # do i need to address packing here, 0 will be deleted anyways, # Warning this assumes (#batchsize, #num_tokens, #embedding_dim), while it may be (#bs, #e_d, #n_t)
            r_q = mxp(S_q) #(100, 1, 200)
            w_emb = qp_bilstm.forward(contexts, sentence_lengths=c_len) #Can we reuse the previous c_representations?
            cwe = common_word_encodings

            # Concatenation operation
            print(w_emb.shape, cwe.shape)
            R_p = torch.cat((w_emb, cwe), 2)
            # Check the following with Stalin!
            R_p = torch.cat((R_p, cwe.extend(batch_size, -1, 200)), 2)
            print(R_p.shape)
            print(w_emb.shape, r_q.shape, cwe.shape)
            input()
            # Recshape operation

            #todo word embeddings
            #todo question independent representation


            # Candidate Representation




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
