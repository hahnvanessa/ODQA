import torch
from torch import nn
import torch.nn.functional as F

class Candidate_Scorer(nn.Module):
	def __init__(self):
		super(Candidate_Scorer, self).__init__()
		self.wb = nn.Linear(200, 1, bias=False) #1 is the output shape because we want one score as output
		self.we = nn.Linear(200, 1, bias=False)


	def begin_scores(self, G_p):
		return self.wb(G_p)

	def end_scores(self, G_p):
		return self.we(G_p)

	def candidate_probabilities(self, G_p, k, pretraining=False):
		'''
		Returns the start and end indices of the top k candidates within a single
		context.
		'''
		b_P = self.begin_scores(G_p) #target vector 1x100
		e_P = self.end_scores(G_p) #target vector 1x100
		numerator = torch.exp(b_P + e_P.transpose(0,1))
		if pretraining:
			candidate_probs = numerator 
		else:
			denominator = torch.sum(numerator)
			candidate_probs = torch.div(numerator, denominator)
		#get top k candidate indices
		upper_diagonal = torch.triu(candidate_probs) #set scores in lower triangular matrix to zero
		H, W = upper_diagonal.shape
		flattened = upper_diagonal.view(-1) #flatten to get top k scores of entire tensor
		k_max_values, flattened_indices = flattened.topk(k)
		orig_indices = torch.cat(((flattened_indices // W).unsqueeze(1), (flattened_indices  % W).unsqueeze(1)), dim=1) #get indices from original, not flattened tensor
		return orig_indices, k_max_values


