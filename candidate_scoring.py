import torch
from torch import nn

class Candidate_Scorer():
    def __init__(self, G_p):
    	self.wb = nn.Linear(G_p.shape)
    	self.we = nn.Linear(G_p.shape)
    	self.G_p = G_p

    def begin_scores(self):
    	return torch.bmm(self.wb.weight, self.G_p) 

    def end_scores(self):
    	return torch.bmm(self.we.weight, self.G_p) 

    def candidate_probabilities(self):
    	b_P = begin_scores(self)
    	e_P = end_scores(self)
    	numerator = torch.exp(torch.add(b_P, e_P))
    	denominator = torch.sum(numerator)
        return torch.div(numerator,denominator)
#TODO: Call the candidate_probabilities in main.py
#TODO: Paper says 'In this definition, the probabilities of all the valid answer candidates are already normalized.'
#		-> need to normalize probabilities?