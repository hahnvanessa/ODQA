import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim, batch_size, dropout=0.2):
        
        super(BiLSTM, self).__init__()

        #what about cuda? where do we need to specify GPU?
        #might need to add more parameters as we will have more features in later stages - advanced representations
        self.embeddings = embedding_matrix
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.embeddings))
        self.dropout = dropout #try without dropout too and with different p
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout, bidirectional=True)
        self.maxpool = nn.MaxPool1d(200) #kernel size is length of sequence

    def forward(self, sentence, sentence_lengths):
        packed_x = pack(self.embedding(sentence), sentence_lengths, batch_first=True, enforce_sorted=False)
        #packed_x = packed_x.view(self.batch_size, self.hidden_dim, self.embedding_dim) # https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
        lstm_out, _ = self.bilstm(packed_x)
        lstm_out, lstm_out_lengths = unpack(lstm_out, total_length=100)
        return lstm_out

    def max_pooling(self, input_tensor):
        r_q = self.maxpool(input_tensor)
        return r_q



def attention(question, context):
    # assuming that input for question and context has dim 54x300 respectively and not 54x1x300
    print(question.shape, context.shape)
    numerator = torch.exp(torch.bmm(question, torch.transpose(context, 1, 2)))
    denominator = torch.sum(numerator) #index 0?
    alpha_tk = torch.div(numerator,denominator) #->dim 54,54

    h_tP = torch.bmm(alpha_tk, question) #->dim 1,300

    #H_P = torch.cat(, dim=0) #dim -> 54,300
    #is h_tP already H_P?
    H_P = h_tP # for now
    return H_P

