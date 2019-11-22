import torch
import sys
sys.path.append('../')

from losses import *
from optimizers import *
from utils.dataset import Sido,Covertype

covertype = Covertype()
X = covertype.X
y = covertype.y


optimizer = ProxSVRGOptimizer()
hp = dict()
hp['max_iter'] = 100
hp['lr'] = 1e-5
hp['s'] = 10
hp['m'] = 10000
hp['eta'] = 1e-4
hp['coeff'] = {'l1':0.0001, 'l2':0.0001}
optimizer.optimize(X,y,hp,log_loss,elastic_net_regularizer,prox_loss)