from argparse import ArgumentParser
from BILSTM import BiLSTM, attention
import os
import pickle
from torch import nn
import torch

def get_file_paths(data_dir):
    # Get paths for all files in the given directory

    file_names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(args.data):
        for file in f:
            if '.pkl' in file:
                file_names.append(os.path.join(r, file))
    return file_names


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
