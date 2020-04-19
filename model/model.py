# torch
import torch
import torch.nn as nn
import torch.nn.functional as 
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
# utils
from utils.BILSTM import BiLSTM, attention, max_pooling
from utils.candidate_scoring import Candidate_Scorer
from utils.candidate_representation import Candidate_Representation


class ODQA(nn.Module):

    def __init__ (self, args):
        super(ODQA, self).__init__()
        self.args = args

        # Constants
        # todo: make this adjustable through args
        self.K = #from args2
        self.MAX_SEQUENCE_LENGTH = #100
        self.batch_size = #

        # Initialize BiLSTMs
        self.qp_bilstm = BiLSTM(embedding_matrix, embedding_dim=300, hidden_dim=100,
                    batch_size=batch_size)
        self.G_bilstm = nn.LSTM(input_size=400, hidden_size=100, bidirectional=True)
        self.sq_bilstm = BiLSTM(embedding_matrix, embedding_dim=300, hidden_dim=100,
                           batch_size=batch_size)  # is embedding dim correct? d_w, #fg: yes
        self.sp_bilstm = nn.LSTM(input_size=501, hidden_size=100, bidirectional=True) #todo: padding function?

        self.fp_bilstm = nn.LSTM(input_size=403,hidden_size=100,bidirectional=True)


        # Candidate Scorer
        self.candidate_scorer = Candidate_Scorer()
        # Candidate Representation
        self.candidate_representation = Candidate_Representation(k=self.K)
        # Answer Scoring
        self.wz = nn.Linear(200, 1, bias=False) # transpose of (1, 200)

    def extract_candidates(self, questions, contexts):
        '''
        Extracts the k candidates with the highest probability from each passage and returns their spans within
        their respective passage.
        '''
        # Question and Passage Representation
        q_representation = self.qp_bilstm.forward(questions, sentence_lengths=q_len) #[100, 100, 200]
        c_representation = self.qp_bilstm.forward(contexts, sentence_lengths=c_len) #[100, 100, 200]
        # Question and Passage Interaction
        HP_attention = attention(q_representation, c_representation) #[100, 100, 200]
        G_input = torch.cat((c_representation, HP_attention), 2)
        G_ps, _ = self.G_bilstm.forward(G_input)
        C_spans = []  # (100x2x2)
        for G_p in G_ps:
            # Store the spans of the top k candidates in the passage
             C_spans.append(self.candidate_scorer.candidate_probabilities(G_p=G_p, k=self.K))  # candidate scores for current context
        C_spans = torch.stack(C_spans, dim=0) #[100, 2, 2]
        return C_spans
    
    
    def get_distance(self, passages, candidates):
        passage_distances = []
        length = candidates.shape[0]
        for i in range(length):
            position_distances = []
            for p in range(passages.shape[1]):
                position_distances.append(torch.dist(passages[i,p,:], candidates[i,:,:]))
            position_distances = torch.stack(position_distances, dim=0)
            passage_distances.append(position_distances.view(1,passages.shape[1]))
        return torch.squeeze(torch.stack(passage_distances, dim=0))


    def represent_passage(self, questions, contexts, common_word_encodings, c_len):
        '''
        Creates an passage representation that is dependent on the question that passage is linked to.
        '''
        S_q = self.sq_bilstm.forward(questions, sentence_lengths=q_len)
        r_q = max_pooling(S_q, self.MAX_SEQUENCE_LENGTH) #(100, 1, 200)
        # Passage Representation
        w_emb = self.qp_bilstm.embed(contexts) # word embeddings (100,100,300)
        R_p = torch.cat((w_emb, common_word_encodings), 2)
        R_p = torch.cat((R_p, r_q.expand(self.batch_size, self.MAX_SEQUENCE_LENGTH, 200)), 2) #(100,100,501)
        packed_R_p = pack(R_p, c_len, batch_first=True, enforce_sorted=False)
        S_p, _ = self.sp_bilstm.forward(packed_R_p)
        S_p, _ = unpack(S_p, total_length=self.MAX_SEQUENCE_LENGTH)  #(100,100,200)
        return S_p



    def compute_passage_advances_repr(self, S_p, S_Cs, r_Cs, r_Ctilde):

        S_P = torch.stack([S_p,S_p],dim=1).view(200,100,200) #reshape S_p
        S_P_attention = attention(S_Cs, S_P) #[200,100,200]
        U_p = torch.cat((S_P, S_P_attention), 2) #[200, 100, 400]
        S_ps_distance =  self.get_distance(S_P,S_Cs)
        U_p = torch.cat((U_p, S_ps_distance.view((200,100,1))), 2)
        print('UP', U_p.shape)
        U_p = torch.cat((U_p, r_Cs.view((200,100,1))), 2) 
        print('UP', U_p.shape)
        U_p = torch.cat((U_p, r_Ctilde.view((200,100,1))), 2) 
        print('UP', U_p.shape)
        packed_U_p = pack(U_p, c_len, batch_first=True, enforce_sorted=False)
        F_p, _ = fp_bilstm.forward(packed_U_p)
        F_p, _ = unpack(F_p, total_length=MAX_SEQUENCE_LENGTH)
        print('FP', F_p.shape)

        return F_p


    def score_answers(self, F_p):

        z_C = max_pooling(F_p, MAX_SEQUENCE_LENGTH)
        s = []
        for c in z_C:
            s.append(wz(c)) # wz:(200,100)
        s = torch.stack(s, dim=0)
        p_C = F.softmax(s, dim=0)

        return p_C

    def forward(self, batch):
        questions, contexts, answers, q_len, c_len, a_len, q_id, common_word_encodings = batch
        # Extract candidate spans form the passages
        C_spans = self.extract_candidates(questions, contexts)
        # Select the candidate that most likely represents the answer
        S_p = self.represent_passage(questions, contexts, common_word_encodings, c_len)

        C_rep = self.candidate_representation.calculate_candidate_representations(S_p=S_p, spans=C_spans, passages=contexts)
        # C_rep = Candidate_Representation(S_p=S_p, spans=C_spans, passages=contexts, k=self.K)
        S_Cs = C_rep.S_Cs  # [200, 100, 200]
        r_Cs = C_rep.r_Cs  # [200, 100]
        r_Ctilde = C_rep.tilda_r_Cs  # [200, 100]
        encoded_candidates = C_rep.encoded_candidates

        F_p = compute_passage_advances_repr(S_p, S_Cs, r_Cs, r_Ctilde)

        answer_probs = score_answers(F_p)
