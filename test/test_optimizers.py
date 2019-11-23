import torch
import sys
sys.path.append('../')

from losses import *
from optimizers import *
from utils.dataset import Sido,Covertype,Rcv1
num_features = 50
num_examples = 1000
sido = Sido()
# X = sido.X
# y = sido.y
X = torch.narrow(sido.X,dim=1,start=0,length=num_features)
X = torch.narrow(X,dim=0,start=0,length=num_examples)
print(X.size())
y = torch.narrow(sido.y,dim=0,start=0,length=num_examples)
num_examples = X.size(0)

optimizer = ProxSVRGOptimizer()
hp = dict()
hp['max_iter'] = 20
hp['lr'] = 5*1e-1
hp['s'] = 150
hp['m'] = num_examples*3
hp['eta'] = 1e-3
hp['coeff'] = {'l1':1e-4, 'l2':1e-8}
optimizer.optimize(X,y,hp,mse,elastic_net_regularizer,prox_loss)


# optimizer = Optimizer()

# X = torch.eye(num_features)
# y = 0.1 * torch.randn(num_features).double()
# weight = optimizer.optimize(X,y,hp,prox_loss)

# print(y)
# print(weight)