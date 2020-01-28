import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, dropout=0.2):
    	#what about cuda? where do we need to specify GPU?
    	#might need to add more parameters as we will have more features in later stages - advanced representations
    	self.hidden_dim = hidden_dim
    	self.batch_size = batch_size
    	self.embedding = #insert glove embedding matrix
    	self.dropout = nn.Dropout(p=dropout) #try without dropout too and with different p
    	self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout, bidirectional=True)

    	self.hidden2label = nn.Linear(hidden_dim, ?) #define second dimension - target length k?

    def forward(self, sentence, attention=False):
    	x = self.embedding(sentence)
    	lstm_out, (h_n, c_n) = self.bilstm(x) #how to input question AND paragraphs?
        #h_n = tensor containing the hidden state for t = seq_len, c_n = tensor containing the cell state for t = seq_len
    	#y = self.hidden2label(lstm_out)
    	return lstm_out, (h_n, c_n)

    def attention(self):
    	pass

    def _get_lstm_features(self, sentence):
    	#this function to return the last hidden layer as contextual representation of the sentence?
    	#also to return more features when we have to implemented the advanced representations later?
    	pass   
    #create another def attention to use in forward function
    #create main.py to loop over the data points and feed into BILSTM