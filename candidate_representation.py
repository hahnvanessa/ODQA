import torch
from torch import nn


class Candidate_Represenation():
    '''
    Condense and interconnect candidates from different
    passages in order to improve the selection of an answer among them.
    '''

    def __init__(self, S_p, spans, k=2):
        print('Started Candidate Representation.')
        self.S_p = S_p
        self.spans = spans
        self.k = k
        self.wb = torch.randn(1, 200)
        self.we = torch.randn(1, 200)
        # Linear transformations to capture the intensity
        # of each interaction (used for the attention mechanism)
        self.wc = nn.Linear(100, 100)
        self.wo = nn.Linear(100, 100)
        self.wv = nn.Linear(1, 100)
        self.rC = self.condensed_vector_representation()

    def condensed_vector_representation(self):
        '''
        Returns the condensed vector representation of all start and all end tokens.
        :return:
        '''
        start_indices = self.spans[:,:,0]
        end_indices = self.spans[:,:,1]
        sp_cb = []
        sp_ce = []

        for p in range(self.S_p.shape[0]):
            # Appends start tokens
            for i in range(self.k):
                sp_cb.append(self.S_p[p][start_indices[p][i]]) # Candidate Nr. i start
            # Append end tokens
            for i in range(self.k):
                sp_ce.append(self.S_p[p][end_indices[p][i]])

        sp_cb = torch.stack(sp_cb, dim=0) #(200x200)
        sp_ce = torch.stack(sp_ce, dim=0) #(200x200)
        # Calculate r_c
        b = torch.mm(self.wb, sp_cb)
        e = torch.mm(self.we, sp_ce)
        r_c = torch.add(b, e).tanh()
        print(r_c.shape)

        return r_c

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


    def calculate_correlations(self, rc, rcm):
        # Model the interactions via attention mechanism

        # TODO: form a matrix V for all candidates
        # TODO: form rcm (condensed vector representation of all
        # candidates except for the current one (rc))

        c = torch.bmm(self.wc, rc)
        o = torch.bmm(self.wo, rcm)
        V_jm = torch.bmm(self.wv, nn.Tanh(torch.add(c, o)))

        return V_jm
