import torch
import sys
sys.path.append('../')

from losses import *
from optimizers import *
from utils.dataset import Sido,Covertype,Rcv1
num_features = 100
num_examples = 5000
sido = Sido()
# X = sido.X
# y = sido.y
X = torch.narrow(sido.X,dim=1,start=0,length=num_features)
X = torch.narrow(X,dim=0,start=0,length=num_examples)
print(X.size())
y = torch.narrow(sido.y,dim=0,start=0,length=num_examples)
num_examples = X.size(0)

optimizer = ProxSGOptimizer()
hp = dict()
hp['max_iter'] = 100
hp['lr'] = 0.1
hp['s'] = 10
hp['m'] = num_examples*20
hp['eta'] = 1e-2
hp['coeff'] = {'l1':1e-4, 'l2':1e-4}
optimizer.optimize(X,y,hp,mse,elastic_net_regularizer,prox_loss,dataset = "Sido")
# op

# optimizer = Optimizer()

# X = torch.eye(num_features)
# y = 0.1 * torch.randn(num_features).double()
# weight = optimizer.optimize(X,y,hp,prox_loss)

# print(y)
# print(weight)