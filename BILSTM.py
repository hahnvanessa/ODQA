import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim, batch_size, dropout=0.2):
    	#what about cuda? where do we need to specify GPU?
    	#might need to add more parameters as we will have more features in later stages - advanced representations
    	self.embeddings = embedding_matrix
    	self.embedding_dim = embedding_dim
    	self.hidden_dim = hidden_dim
    	self.batch_size = batch_size
    	self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.embeddings))
    	self.dropout = nn.Dropout(p=dropout) #try without dropout too and with different p
    	self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout, bidirectional=True)

    	#self.hidden2label = nn.Linear(hidden_dim, ?) #define second dimension - target length k?

    def forward(self, sentence, attention=False):
    	x = self.embedding(sentence)
    	lstm_out, (h_n, c_n) = self.bilstm(x) #how to input question AND paragraphs?
        #h_n = tensor containing the hidden state for t = seq_len, c_n = tensor containing the cell state for t = seq_len
    	#y = self.hidden2label(lstm_out)
    	return lstm_out, (h_n, c_n)

    def attention(self, question, context):
        # assuming that input for question and context has dim 54x300 respectively and not 54x1x300 
        numerator = torch.exp(torch.bmm(question, torch.transpose(context, 0, 1)))
        denominator = torch.sum(numerator) #index 0?
        alpha_tk = torch.div(numerator,denominator) #->dim 54,54

        h_tP = torch.sum(torch.bmm(alpha_tk, question)) #->dim 1,300

        #H_P = torch.cat(, dim=0) #dim -> 54,300
        #is h_tP already H_P?
    	H_P = h_tP # for now 
        return H_P
 
    #create another def attention to use in forward function
    #create main.py to loop over the data points and feed into BILSTM