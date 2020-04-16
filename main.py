from argparse import ArgumentParser
from BILSTM import BiLSTM, attention, max_pooling
import os
import pickle
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import question_answer_set
from torch.utils.data import Dataset, DataLoader
import question_answer_set as qas
import candidate_scoring
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from candidate_representation import Candidate_Representation

MAX_SEQUENCE_LENGTH = 100
K = 2 # Number of extracted candidates per passage

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
    sq_bilstm = BiLSTM(embedding_matrix, embedding_dim=300, hidden_dim=100,
                       batch_size=batch_size)  # is embedding dim correct? d_w, #fg: yes
    sp_bilstm = nn.LSTM(input_size=501, hidden_size=100, bidirectional=True) #todo: padding function?
    fp_bilstm = nn.LSTM(input_size=403,hidden_size=100,bidirectional=True)


    for epoch in range(num_epochs):

        for batch_number, data in enumerate(train_loader):
            questions, contexts, answers, q_len, c_len, a_len, q_id, common_word_encodings = data
            print('Started candidate extraction...')
            # region Part 1 - Candidate Extraction
            # Question and Passage Representation
            q_representation = qp_bilstm.forward(questions, sentence_lengths=q_len) #[100, 100, 200]
            c_representation = qp_bilstm.forward(contexts, sentence_lengths=c_len) #[100, 100, 200]
            # Question and Passage Interaction
            HP_attention = attention(q_representation, c_representation) #[100, 100, 200]
            G_input = torch.cat((c_representation, HP_attention), 2)
            G_ps, _ = G_bilstm.forward(G_input)
            C_spans = []  # (100x2x2)
            for G_p in G_ps:
                # Store the spans of the top k candidates in the passage
                C_spans.append(candidate_scoring.Candidate_Scorer(G_p).candidate_probabilities(K))  # candidate scores for current context
            C_spans = torch.stack(C_spans, dim=0) #[100, 2, 2]
            # if we create only one candidate scorer instance before (e.g. one for each question or one for all questions), we need to change the G_p argument
            # endregion

            # region Part 2 - Answer Selection
            # Question Representation (Condensed Question)
            print('Started Answer Selection...')
            S_q = sq_bilstm.forward(questions, sentence_lengths=q_len)
            r_q = max_pooling(S_q, MAX_SEQUENCE_LENGTH) #(100, 1, 200)
            # Passage Representation
            w_emb = qp_bilstm.embed(contexts) # word embeddings (100,100,300)
            R_p = torch.cat((w_emb, common_word_encodings), 2)
            R_p = torch.cat((R_p, r_q.expand(batch_size, MAX_SEQUENCE_LENGTH, 200)), 2) #(100,100,501)
            packed_R_p = pack(R_p, c_len, batch_first=True, enforce_sorted=False)
            S_p, _ = sp_bilstm.forward(packed_R_p)
            S_p, _ = unpack(S_p, total_length=MAX_SEQUENCE_LENGTH)  #(100,100,200)
            #print('shapes', S_p.shape, C_spans.shape)

            # Candidate Representation
            # todo: Do we need to share the weights among the multiple Candidate Rep classes? (Same goes for candidate scores?)
            # In that case we would need to make the Candidate Rep functions take inputs e.g.
            # generate_fused_representation(V). Right now the functions take these values directly from the class.
            C_rep = Candidate_Representation(S_p, C_spans, k=K)
            S_Cs = C_rep.S_Cs #[200, 100, 200]
            r_Cs = C_rep.r_Cs #[200, 100]
            r_Ctilde = C_rep.tilda_r_Cs #[200, 100]

            #Passage Advanced Representation
            S_P = torch.stack([S_p,S_p],dim=1).view(200,100,200) #reshape S_p
            S_P_attention = attention(S_Cs, S_P) #[200,100,200] 
            U_p = torch.cat((S_P, S_P_attention), 2) #[200, 100, 400]
            S_ps_distance =  get_distance(S_P,S_Cs)
            print('distance', S_ps_distance.shape)
            U_p = torch.cat((U_p, S_ps_distance.view((200,100,1))), 2)
            print('UP', U_p.shape)
            U_p = torch.cat((U_p, r_Cs.view((200,100,1))), 2) 
            print('UP', U_p.shape)
            U_p = torch.cat((U_p, r_Ctilde.view((200,100,1))), 2) 
            print('UP', U_p.shape)
            packed_U_p = pack(U_p, c_len, batch_first=True, enforce_sorted=False)
            F_p, _ = fp_bilstm.forward(U_p)
            print('FP', F_p.shape)
            F_p, _ = unpack(F_p, total_length=MAX_SEQUENCE_LENGTH)
            print('FP', F_p.shape)

            # endregion

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
