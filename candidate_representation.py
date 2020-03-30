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

 

    def calculate_correlations(self):
        # Model the interactions via attention mechanism

        r_Cs = torch.split(self.rC, 100, dim=0) # TODO: check the dimensions

        V_jms = []

        for i, r_Cs in enumerate(r_Cs):
            rcm = torch.cat([r_Cs[0:i], r_Cs[i+1:]], dim=0)
            c = torch.bmm(self.wc, r_Cs)
            o = torch.bmm(self.wo, rcm)
            V_jm = torch.bmm(self.wv, torch.add(c, o).tahn())
            V_jms.append(V_jm)

        V = torch.stack(V_jms, dim=0)

        return V


    def generate_fused_representation(self, V):

        # Normalize interactions
        V_jms = torch.split(V, ..., dim=0) # TODO: check the dimensions

        alpha_ms = []

        for i, V_jm in enumerate(V_jms):
            numerator = torch.exp(V_jm)
            denominator_correlations = torch.stack([V_jms[0:i], V_jms[i:]], dim=0)
            denominator = torch.sum(torch.exp(denominator_correlations), dim =0)
            alpha_m = torch.div(numerator, denominator)
            alpha_ms.append(alpha_m)

        alpha = torch.stack(alpha_ms, dim=0)


        # Generate fused representations
        r_Cs = torch.split(self.rC, 100, dim=0) # TODO: check the dimensions

        tilda_rсms = []

        for i, r_Cs in enumerate(r_Cs):
            rcm = torch.cat([r_Cs[0:i], r_Cs[i+1:]], dim=0)
            tilda_rсm = torch.bmm(alpha[i], rcm)
            tilda_rcms.append(tilda_rсm)

        tilda_rcms = torch.stack(tilda_rcms, dim=0)
        tilda_rC = torch.sum(tilda_rcms)

        return tilda_rC