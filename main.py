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

MAX_SEQUENCE_LENGTH = 100

def get_file_paths(data_dir):
    # Get paths for all files in the given directory
    file_names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(args.data):
        for file in f:
            if '.pkl' in file:
                file_names.append(os.path.join(r, file))
    return file_names



def batch_training(dataset, embedding_matrix, batch_size=6, num_epochs=10):
    '''
    Performs minibatch training
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

    # Initialize BiLSTMs
    qp_bilstm = BiLSTM(embedding_matrix, embedding_dim=300, hidden_dim=100,
                batch_size=1)
    interaction_bilstm = BiLSTM(embedding_matrix, embedding_dim=400, hidden_dim=100,
                batch_size=1)
    G_bilstm = nn.LSTM(input_size=400, hidden_size=100, bidirectional=True)


    for epoch in range(num_epochs):

        for batch_number, data in enumerate(train_loader):
            questions, contexts, answers, q_len, c_len, a_len = data
            # Pack (reduce the sentences to their original form without padding)
            packed_q = torch.nn.utils.rnn.pack_padded_sequence(questions, q_len, batch_first=True)
            packed_c = torch.nn.utils.rnn.pack_padded_sequence(contexts, c_len, batch_first=True)
            packed_a = torch.nn.utils.rnn.pack_padded_sequence(answers, a_len, batch_first=True)
            # Forward operations
            q_representation = qp_bilstm.forward(packed_q)
            c_representation = qp_bilstm.forward(packed_c)
            HP_attention = attention(q_representation, c_representation)
            # todo: finish the forward operations, this is basically just transferring stuff from the main function into here

            # Unpack (add the paddings again, so the sentences are in their original form)
            G_p_packed = torch.nn.utils.rnn.pad_packed_sequence(G_p_packed, batch_first=True)
            G_ps.append(G_p_packed)




def main(embedding_matrix, encoded_corpora):
    '''
    Iterates through all given corpus files and forwards the encoded contexts and questions
    through the BILSTMs.
    :param embedding_matrix:
    :param encoded_corpora:
    :return:
    '''

    embedding_matrix = pickle.load(open(embedding_matrix, 'rb'))

    # Create BILSTMs
    qp_bilstm = BiLSTM(embedding_matrix, embedding_dim=300, hidden_dim=100,
                batch_size=1)
    interaction_bilstm = BiLSTM(embedding_matrix, embedding_dim=400, hidden_dim=100,
                batch_size=1)
    G_bilstm = nn.LSTM(input_size=400, hidden_size=100, bidirectional=True)

    # Retrieve the filepaths of all encoded corpora
    file_paths = get_file_paths(encoded_corpora)

    qp_representations = {}
    int_representations = {}

    for file in file_paths:
        with open(os.path.join(file), 'rb') as f:
            content = pickle.load(f)

            # Minibatch training
            dataset = qas.question_answer_set(content)
            batch_training(dataset, embedding_matrix, batch_size=6, num_epochs=10)

            # todo: transfer all this code into batch_training()
            '''
            for item in content:
                item_id = item
                question = content[item_id]['encoded_question']
                question = torch.tensor(question).to(torch.int64)
                q_representation = qp_bilstm.forward(question)  # get the question representation
                contexts = content[item_id]['encoded_contexts']
                c_representations = []
                G_ps = []

                for context in contexts:
                    context = torch.tensor(context).to(torch.int64)
                    c_representation = qp_bilstm.forward(context)  # get the context representation
                    c_representations.append(c_representation)
                    HP_attention = attention(q_representation, c_representation)
                    G_input = torch.cat((c_representation, HP_attention), 2)
                    G_p = G_bilstm.forward(G_input)
                    G_ps.append(G_p)
                int_representations[item_id] = G_ps
                qp_representations[item_id] = {'q_repr': q_representation,
                                               'c_repr': c_representations}
            '''
if __name__ == '__main__':

    parser = ArgumentParser(
        description='Main ODQA script')
    parser.add_argument(
        'embeddings', help='Path to the pkl file')
    parser.add_argument(
        'data', help='Path to the folder with the pkl files')

    # Parse given arguments
    args = parser.parse_args()

    # Call main()
    main(embedding_matrix=args.embeddings, encoded_corpora=args.data)
