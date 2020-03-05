import torch

class Loss_Function():
    '''
    Contains all function necessary to calculate the loss
    '''

    def __init__(self):
        pass

    def loss(self):
        pass

def reward(c, a, c_len, a_len):
    '''
    Returns the reward for a candidate prediction.
    todo: Make this capable of using batches
    :param candidate:
    :param answer:
    :return:
    '''

    def f1_score(c, a, c_len, a_len, epsilon=1e-7):
        # source for reference: https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
        y_true = torch.ones(c.shape, dtype=torch.bool).long()
        y_pred = torch.eq(c, a).long()
        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        return f1

    # Trim padding to candidate len
    assert c_len >= a_len
    c = c[0, :c_len]
    a = a[0, :c_len]

    if torch.all(torch.eq(c, a)):
        return 2
    elif len(torch.eq(c, a).nonzero()) > 0:
        return f1_score(c, a, c_len, a_len)
    else:
        return -1