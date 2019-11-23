import torch
import sys
sys.path.append('../')

from losses import *
from optimizers import *
from utils.dataset import Sido,Covertype
num_features = 50
num_examples = 1000
sido = Sido()
X = torch.narrow(sido.X,dim=1,start=0,length=num_features)
X = torch.narrow(X,dim=0,start=0,length=num_examples)
print(X.size())
y = torch.narrow(sido.y,dim=0,start=0,length=num_examples)


optimizer = ProxSAGOptimizer()
hp = dict()
hp['max_iter'] = 500
hp['lr'] = 1e-5
hp['s'] = 50
hp['m'] = 50000
hp['eta'] = 0.5*1e-4
hp['coeff'] = {'l1':1e-4, 'l2':1e-4}
optimizer.optimize(X,y,hp,log_loss,elastic_net_regularizer,prox_loss)