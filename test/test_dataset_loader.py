import sys
sys.path.append('../utils')

import torch
from dataset import Rcv1, Covertype, Sido

rcv1 = Rcv1()
print('Loaded Rcv1')
covertype = Covertype()
print('Loaded Covertype')
sido = Sido()
print('Loaded Sido')
