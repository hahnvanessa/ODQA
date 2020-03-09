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
        b_P = self.begin_scores() #target vector 1x100
        e_P = self.end_scores() #target vector 1x100
        # todo: perform this as a matrix addition
        numerator = torch.exp(b_P + e_P.transpose(0,1))
        denominator = torch.sum(numerator) 
        x = torch.div(numerator, denominator)
        norm = x.norm(p=2, dim=1, keepdim=True)
        x_normalized = x.div(norm)
        return x_normalized
    #return torch.div(numerator,denominator) #unnormalized
    #return torch.norm(torch.div(numerator,denominator)) #normalized
