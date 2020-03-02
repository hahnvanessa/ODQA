import torch
from torch import nn

class Candidate_Scorer():
	def __init__(self, G_p):
		self.wb = nn.Linear(200, 1) #1 is the output shape because we want one score as output
		self.we = nn.Linear(200, 1)
		self.G_p = G_p

	def begin_scores(self):
		return self.wb(self.G_p) 

	def end_scores(self):
		return self.we(self.G_p) 

	def candidate_probabilities(self):
		b_P = self.begin_scores()
		e_P = self.end_scores()
		numerator = torch.exp(torch.add(b_P, e_P))
		denominator = torch.sum(numerator)
		#return torch.div(numerator,denominator) #unnormalized
		return torch.norm(torch.div(numerator,denominator)) #normalized
