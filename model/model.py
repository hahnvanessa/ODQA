# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
# utils
from utils.BILSTM import BiLSTM, attention, max_pooling
from utils.candidate_scoring import Candidate_Scorer
from utils.candidate_representation import Candidate_Representation


class ODQA(nn.Module):

    def __init__ (self, k, max_sequence_length, batch_size, embedding_matrix, device)):
        super(ODQA, self).__init__()
        # Constants
        # todo: make this adjustable through args
        self.K = k
        self.MAX_SEQUENCE_LENGTH = max_sequence_length
        self.BATCH_SIZE = batch_size
        self.device = device

        # Initialize BiLSTMs
        self.qp_bilstm = BiLSTM(embedding_matrix, embedding_dim=300, hidden_dim=100)
        self.G_bilstm = nn.LSTM(input_size=400, hidden_size=100, bidirectional=True)
        self.sq_bilstm = BiLSTM(embedding_matrix, embedding_dim=300, hidden_dim=100)  # is embedding dim correct? d_w, #fg: yes
        self.sp_bilstm = nn.LSTM(input_size=501, hidden_size=100, bidirectional=True) #todo: padding function?
        self.fp_bilstm = nn.LSTM(input_size=403,hidden_size=100,bidirectional=True)


        # Initialize Candidate Objects
        self.candidate_scorer = Candidate_Scorer()
        # Candidate Representation
        self.candidate_representation = Candidate_Representation(k=self.K)
        # Answer Scoring
        self.wz = nn.Linear(200, 1, bias=False) # transpose of (1, 200)

    def reset_batch_size(self, batch_size):
        self.BATCH_SIZE = batch_size

    def store_parameters(self, filepath):
        # A common PyTorch convention is to save models using either a .pt or .pth file extension.
        torch.save(self.state_dict(), filepath)
        print('Stored parameters')

    def load_parameters(self, filepath):
        self.load_state_dict(torch.load(filepath))
        print('Retrieved parameters')


    def extract_candidates(self, questions, contexts, q_len, c_len):
        '''
        Extracts the k candidates with the highest probability from each passage and returns their spans within
        their respective passages.
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


    def compute_passage_representation(self, questions, contexts, common_word_encodings, q_len, c_len):
        '''
        Computes a passage representation that is dependent on the question that passage is linked to.
        '''
        S_q = self.sq_bilstm.forward(questions, sentence_lengths=q_len)
        r_q = max_pooling(S_q, self.MAX_SEQUENCE_LENGTH) #(100, 1, 200)
        # Passage Representation
        w_emb = self.qp_bilstm.embed(contexts) # word embeddings (100,100,300)
        R_p = torch.cat((w_emb, common_word_encodings), 2)
        R_p = torch.cat((R_p, r_q.expand(self.BATCH_SIZE, self.MAX_SEQUENCE_LENGTH, 200)), 2) #(100,100,501)
        packed_R_p = pack(R_p, c_len, batch_first=True, enforce_sorted=False)
        S_p, _ = self.sp_bilstm.forward(packed_R_p)
        S_p, _ = unpack(S_p, total_length=self.MAX_SEQUENCE_LENGTH)  #(100,100,200)
        return S_p



    def compute_passage_advanced_representation(self, c_len, S_p, S_Cs, r_Cs, r_Ctilde):
        S_P = torch.stack([S_p,S_p],dim=1).view(200,100,200) #reshape S_p
        S_P_attention = attention(S_Cs, S_P) #[200,100,200]
        U_p = torch.cat((S_P, S_P_attention), 2) #[200, 100, 400]
        S_ps_distance =  self.get_distance(S_P,S_Cs)
        U_p = torch.cat((U_p, S_ps_distance.view((200,100,1))), 2)
        U_p = torch.cat((U_p, r_Cs.view((200,100,1))), 2)
        U_p = torch.cat((U_p, r_Ctilde.view((200,100,1))), 2)
        packed_U_p = pack(U_p, c_len, batch_first=True, enforce_sorted=False)
        F_p, _ = self.fp_bilstm.forward(packed_U_p)
        F_p, _ = unpack(F_p, total_length=self.MAX_SEQUENCE_LENGTH)
        return F_p


    def score_answers(self, F_p):
        z_C = max_pooling(F_p, self.MAX_SEQUENCE_LENGTH)
        s = []
        for c in z_C:
            s.append(self.wz(c)) # wz:(200,100)
        s = torch.stack(s, dim=0)
        p_C = torch.softmax(s, dim=0)

        return p_C

    def forward(self, batch):
        questions, contexts, gt_contexts, answers, q_len, c_len, a_len, q_id, common_word_encodings = batch
        # Feed to GPU
        questions = questions.to(self.device)
        contexts = contexts.to(self.device)
        gt_contexts = gt_contexts.to(self.device)	
        answers = answers.to(self.device)
        q_len = q_len.to(self.device)
        c_len = c_len.to(self.device)
        a_len = a_len.to(self.device)
        q_id  = q_id.to(self.device)
        common_word_encodings = common_word_encodings.to(self.device)

        # Extract candidate spans form the passages
        C_spans = self.extract_candidates(questions, contexts, q_len, c_len)
        # Represents the passage as being dependent on the answer
        S_p = self.compute_passage_representation(questions, contexts, common_word_encodings, q_len=q_len, c_len=c_len)
        # Represents the candidates
        self.candidate_representation.calculate_candidate_representations(S_p=S_p, spans=C_spans, passages=contexts)
        S_Cs = self.candidate_representation.S_Cs  # [200, 100, 200]
        r_Cs = self.candidate_representation.r_Cs  # [200, 100]
        r_Ctilde = self.candidate_representation.tilda_r_Cs  # [200, 100]
        encoded_candidates = self.candidate_representation.encoded_candidates
        # Compute an advanced representation of the passage
        F_p = self.compute_passage_advanced_representation(c_len=c_len, S_p=S_p, S_Cs=S_Cs, r_Cs=r_Cs, r_Ctilde= r_Ctilde)
        # Commpute the probabilities of the candidates (highest should be the ground truth answer)
        p_C = self.score_answers(F_p)
        # Return the embedding-index version of the candidate with the highest probability
        # todo: check if this works and if this always returns only one  value
        value, index = torch.max(p_C, 0)
        # todo: Maybe we can use the value to find out how certain the algorithm is about our candidate
        # todo: returns only one answer for all the datapoints
        return encoded_candidates[index][0][0], questions[0], answers[0]
