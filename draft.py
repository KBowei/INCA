import torch

x = torch.ones(4, 4, 2, 2)
y = torch.cat([x]*4, dim=0)
print(y.shape)
LL, HL, LH, HH = torch.chunk(x, 4, dim=1)
print(LL.shape)
