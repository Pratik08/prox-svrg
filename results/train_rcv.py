'''
Authors: Pratik Dubal, Shashwat Verma and Saurabh Sharma.

All the code is vectorized and cuda enabled.
'''

import torch
import pickle
import sys
sys.path.append('../')

from losses import *
from optimizers import *
from utils.dataset import Rcv1

##################################
# Rcv1 dataset
num_features = 1000
num_examples = 5000
dset = Rcv1()

# Choosing subset of dataset
X = torch.narrow(dset.X, dim=1, start=0, length=num_features)
X = torch.narrow(X, dim=0, start=0, length=num_examples)
y = torch.narrow(dset.y, dim=0, start=0, length=num_examples)
num_examples = X.size(0)
##################################
# Full Gradient Descent
optimizer = Optimizer()
hp = dict()
hp['max_iter'] = 100
hp['lr'] = 0.1
hp['s'] = 10
hp['m'] = num_examples*20
hp['eta'] = 1e-2
hp['coeff'] = {'l1': 1e-4, 'l2': 1e-4}
optimizer.optimize(X, y, hp, log_loss, elastic_net_regularizer, prox_loss, dataset="rcv")
with open('rcv_gd_nnz.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.num_non_zeros, f)

with open('rcv_gd_efp.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.effective_passes, f)

with open('rcv_gd_og.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.objective_gap, f)
##################################
# Prox-SG Descent
optimizer = ProxSGOptimizer()
hp = dict()
hp['max_iter'] = 100
hp['lr'] = 0.1
hp['s'] = 10
hp['m'] = num_examples*20
hp['eta'] = 1e-2
hp['coeff'] = {'l1':1e-4, 'l2':1e-4}
optimizer.optimize(X, y, hp, log_loss, elastic_net_regularizer, prox_loss, dataset="rcv")
with open('rcv_proxsg_nnz.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.num_non_zeros, f)

with open('rcv_proxsg_efp.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.effective_passes, f)

with open('rcv_proxsg_og.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.objective_gap, f)
##################################
# Prox-SVRG Descent
optimizer = ProxSVRGOptimizer()
hp = dict()
hp['max_iter'] = 100
hp['lr'] = 0.1
hp['s'] = 10
hp['m'] = num_examples*20
hp['eta'] = 1e-2
hp['coeff'] = {'l1':1e-4, 'l2':1e-4}
optimizer.optimize(X, y, hp, log_loss, elastic_net_regularizer, prox_loss, dataset="rcv")
with open('rcv_proxsvrg_nnz.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.num_non_zeros, f)

with open('rcv_proxsvrg_efp.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.effective_passes, f)

with open('rcv_proxsvrg_og.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.objective_gap, f)
##################################
# Prox-SAG Descent
optimizer = ProxSAGOptimizer()
hp = dict()
hp['max_iter'] = 100
hp['lr'] = 0.1
hp['s'] = 10
hp['m'] = num_examples*20
hp['eta'] = 1e-2
hp['coeff'] = {'l1':1e-4, 'l2':1e-4}
optimizer.optimize(X, y, hp, log_loss, elastic_net_regularizer, prox_loss, dataset="rcv")
with open('rcv_proxsag_nnz.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.num_non_zeros, f)

with open('rcv_proxsag_efp.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.effective_passes, f)

with open('rcv_proxsag_og.pkl', 'wb') as f:
    pickle.dump(optimizer.stats.objective_gap, f)
##################################
