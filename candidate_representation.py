import torch
from torch import nn


class Candidate_Represenation():

    def __init__(self, S_p, scores):
        self.S_p = S_p
        self.scores = scores
        self.wb = torch.randn(100, 200)
        self.we = torch.randn(100, 200)
        # Linear transformations to capture the intensity
        # of each interaction (used for the attention mechanism)
        self.wc = torch.Linear(100, 100)
        self.wo = torch.Linear(100, 100)
        self.wv = torch.Linear(1, 100)

    def extract_spans(self):

        # Extract candidate spans from the passage representation
        # TODO: iterate over all the passages in the main.py
        
        # Get indicies only for nonzero elements of the tensor
        indicies = self.scores.nonzero()
        
        S_c = []
        for row in indicies:
            sp_b = self.S_p(row[0])
            sp_e = self.S_p(row[1])
            s_c = torch.stack([sp_b, sp_e], dim=0) # 2 x 200
            S_c.append(s_c)
        S_c = torch.stack(S_c, dim=0) # K(100) x 2 x 200

        return S_c


    def compute_rc(self):
        # TODO: iterate over all the entries in the main.py
        
        # Compute codensed vector representation
        b = torch.bmm(self.wb, self.sp_b)
        e = torch.bmm(self.we, self.sp_e)
        r_c = nn.Tanh(torch.add(b, e))

        return r_c


    def calculate_correlations(self):
        # Model the interactions via attention mechanism

        r_Cs = torch.split(self.rC, 100, dim=0) # TODO: check the dimensions

        V_jms = []

        for i, r_Cs in enumerate(r_Cs):
            rcm = torch.cat([r_Cs[0:i], r_Cs[i+1:]], dim=1)
            c = torch.bmm(self.wc, r_Cs)
            o = torch.bmm(self.wo, rcm)
            V_jm = torch.bmm(self.wv, torch.add(c, o).tahn())
            V_jms.append(V_jm)

        V = torch.stack(V_jms, dim=0)

        return V