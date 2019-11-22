import torch
import sys
sys.path.append('../')

from losses import mse
from optimizers import Optimizer
from utils.dataset import Sido

sido = Sido()
X = sido.X
y = sido.y

optimizer = Optimizer()
hp = dict()
hp['max_iter'] = 5
hp['lr'] = 1e-3
optimizer.optimize(X,y,hp,mse)