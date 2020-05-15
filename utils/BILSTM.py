import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim, dropout=0.1):
        super(BiLSTM, self).__init__()

        self.embeddings = embedding_matrix
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.embeddings))
        self.dropout = dropout
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout, bidirectional=True)

    def embed(self, sentence):
        return self.embedding(sentence)

    def forward(self, sentence, sentence_lengths):
        packed_x = pack(self.embedding(sentence), sentence_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed_x)
        lstm_out, lstm_out_lengths = unpack(lstm_out, total_length=100, batch_first=True)
        return lstm_out

def max_pooling(input_tensor, max_sequence_length):

    mxp = nn.MaxPool2d((max_sequence_length, 1),stride=1)
    return mxp(input_tensor)

def attention(questions, contexts):
    max_value = torch.max(torch.max(torch.bmm(questions, torch.transpose(contexts, 1, 2))))
    numerator = torch.exp(torch.bmm(questions, torch.transpose(contexts, 1, 2)) - max_value)
    denominator = torch.sum(numerator)

    alpha_tk = torch.div(numerator,denominator)
    

    h_tP = torch.bmm(alpha_tk, questions)

    H_P = h_tP 
 
    return H_P
    
