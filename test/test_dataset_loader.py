import sys
sys.path.append('../utils')

import torch
from dataset import Rcv1, Sido

rcv1 = Rcv1()
print('Loaded Rcv1')
sido = Sido()
print('Loaded Sido')
