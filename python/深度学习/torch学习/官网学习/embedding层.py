import torch
from torch import nn

embedding = nn.Embedding(10, 3)
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(embedding(input))

embedding = nn.Embedding(6, 3)
input = torch.LongTensor([[1],[2],[3],[4],[5]])
print(embedding(input))



# embedding = nn.Embedding(10, 3, padding_idx=0)
# input = torch.LongTensor([[0,2,0,5]])
# embedding(input)
#
#
# padding_idx = 0
# embedding = nn.Embedding(3, 3, padding_idx=padding_idx)

