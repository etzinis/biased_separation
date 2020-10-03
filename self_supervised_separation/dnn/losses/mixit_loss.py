import torch
import torch.nn as nn
import itertools
import numpy as np


# loss = 10log10(|y-y'|^2 + SNR_max * |y|^2) - 10log10(|y|^2)
# negative threshold SNR
def LossSNR(y, y_pred, SNR_max=30):
     # y: (bs, t)
     # y_pred: (bs, t)
    squared_error = torch.sum((y-y_pred)**2, axis=1)
    y_norm_sqr = torch.sum(y**2, axis=1)
    loss = 10 * torch.log10(squared_error + SNR_max * y_norm_sqr) - 10 * torch.log10(y_norm_sqr)
    return torch.sum(loss)



class MixITLoss(nn.Module):
    def __init__(self, max_num_sources_per_mix):

        max_num_out_sources = 2 * max_num_sources_per_mix
        
        # Mixit matrix, allow two mixtures to have different # of sources
        self.A_tensor = torch.zeros((2**max_num_out_sources, 2, max_num_out_sources))
        
        self.num_combinations = 2**max_num_out_sources
        combinations = torch.tensor(list(itertools.product([0, 1], repeat=max_num_out_sources)))
        for i in range(self.num_combinations):
            self.A_tensor[i,0] = combinations[i]
            self.A_tensor[i,1] = 1 - combinations[i]
    
    def forward(self, estimated_sources, m1, m2):
        # this function returns the minimum loss and the index of the corresponding mixit matrix 
        bs = estimated_sources.shape[0]
        time_dim = m1.shape[-1]
        losses = []

        for idx_iter in range(self.num_combinations):

            # matmul broadcast A_tensor[idx_iter]
            estimated_mixtures = torch.matmul(self.A_tensor[idx_iter], estimated_sources)
            
            # shape of estimated_mixtures(bs, 2, t)
            this_loss = LossSNR(estimated_mixtures[:, 0], m1) + LossSNR(estimated_mixtures[:, 1], m2)
            losses.append(this_loss)

        return torch.argmin(torch.tensor(losses)), torch.tensor(min(losses))


def test_mixit_loss():
    y = torch.tensor([[1.,2.,3.,4.], [2.,3.,4.,5.], [1.,2.,3.,4.], [2.,3.,4.,5.]])
    print(LossSNR(y, y))
    y_pred = torch.tensor([[2.,3.,4.,5.], [3.,4.,5.,6.], [2.,3.,4.,5.], [3.,4.,5.,6.]])
    print(LossSNR(y, y_pred))

    loss = MixITLoss(2)
    print(loss.A_tensor)

    m1 = torch.tensor([3., 1., 7., 0.]).reshape(1, 4)
    m2 = torch.tensor([3., 3., 4., 0.]).reshape(1, 4)
    print(m1.shape)

    time_dim = 4

    estimated_sources = torch.tensor([[1., 0., 2., 0.], [0., 1., 3., 0.], [2., 1., 5., 0.], [3., 2., 1., 0.]]).reshape(1, 4, 4)

    # let forward return index before doing this
    index, min_loss = loss.forward(estimated_sources, m1, m2)

    print(loss.A_tensor[index])
    print(min_loss)

    print(torch.matmul(loss.A_tensor[index], estimated_sources)) # this should equal to m1 m2

if __name__ == "__main__":
    test_mixit_loss()
    pass