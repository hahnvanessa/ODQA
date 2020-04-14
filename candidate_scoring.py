import torch
from torch import nn
import torch.nn.functional as F

class Candidate_Scorer():
	def __init__(self, G_p):
		self.wb = nn.Linear(200, 1) #1 is the output shape because we want one score as output
		self.we = nn.Linear(200, 1)
		self.G_p = G_p

	def begin_scores(self):
		return self.wb(self.G_p)

	def end_scores(self):
		return self.we(self.G_p)

	def candidate_probabilities(self, k):
		'''
		Returns the start and end indices of the top k candidates within a single
		context.
		'''
		b_P = self.begin_scores() #target vector 1x100
		e_P = self.end_scores() #target vector 1x100
		candidate_prob = F.softmax(b_P + e_P, dim=0)
		#get top k candidate indices
		upper_diagonal = torch.triu(candidate_prob) #set scores in lower triangular matrix to zero
		H, W = upper_diagonal.shape
		flattened = upper_diagonal.view(-1) #flatten to get top k scores of entire tensor
		k_max_values, flattened_indices = flattened.topk(k)
		orig_indices = torch.cat(((flattened_indices // W).unsqueeze(1), (flattened_indices  % W).unsqueeze(1)), dim=1) #get indices from original, not flattened tensor
		return orig_indices
