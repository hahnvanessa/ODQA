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
    	#self.hidden2label?

    	self.hidden = self.init_hidden()
    	self.hidden2label = nn.Linear(hidden_dim, ?) #define second dimension - target length k?


    def init_hidden(self):
    	#something like this
    	 return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def forward(self, sentence):
    	x = self.embedding(sentence)
    	lstm_out, self.hidden = self.lstm(x)
    	y = self.hidden2label(lstm_out)
    	return y

    def _get_lstm_features(self, sentence):
    	#this function to return the last hidden layer as contextual representation of the sentence?
    	#also to return more features when we have to implemented the advanced representations later?
    	pass