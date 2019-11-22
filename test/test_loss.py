import torch
import sys
sys.path.append('../')

from losses import mse

X = torch.randn(5, 3)
y = torch.randn(5)
w = torch.randn(3)

print('Compute: ', mse.compute(X, y, w))
print('Grad: ', mse.grad(X, y, w))
