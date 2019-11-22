from copy import deepcopy
from stats import Stats
import torch
import random

'''
	Base class with a Gradient Descent optimizer
	optimize is a function which will be overriden by derived classes. 
	optimize parameters:
		X - data
		y - labels/expected outputs
		hp - dict of hyperparameters
		loss - object with F part of objective function
		regularizer - object with R part of objective function
		prox - object with proximal mapping function
'''
class Optimizer:
	def __init__(self):
		self.stats = Stats()

	def optimize(self, X, y, hp, loss, regularizer=None, prox = None):

		# Gradient 
		w = torch.randn(X.size(1)) * 0.01
		for _ in range(hp['max_iter']):
			grad = loss.grad(X,y,w)
			if regularizer is not None:
				grad = torch.add(grad,regularizer.grad(w, hp['coeff']))
			w = w - hp['lr'] * grad
			self.stats.compute(w, loss.compute(X,y,w))
		# print(self.stats.num_non_zeros)
		# print(self.stats.objective_gap)
		return w



'''
	Hyperparamters: 
		s - number of stages
		m - number of iterations per stage
		eta - learning rate
'''
class ProxSVRGOptimizer(Optimizer):
	def __init__(self):
		super().__init__()


	def optimize(self, X, y, hp, loss, regularizer, prox):
		num_examples, num_params = X.size(0), X.size(1)
		eta, m, s = hp['eta'], hp['m'], hp['s']
		prox_optimizer = Optimizer()
		w_bar = torch.randn(num_params) * 0.1
		for i in range(s):
			v_bar = loss.grad(X, y, w_bar)
			weight_iterates = torch.reshape(w_bar,(1,-1))
			for k in range(1,m + 1):
				q = random.randint(0,num_examples-1)
				p1 = loss.grad(torch.reshape(X[k],(1,-1)),y[k],weight_iterates[k-1])
				p2 = loss.grad(torch.reshape(X[k],(1,-1)),y[k],w_bar)
				v_k = torch.add(torch.sub(p1,p2),v_bar)
				prox_input = torch.sub(weight_iterates[k-1], torch.mul(v_k, eta))
				prox_input = torch.tensor(prox_input,dtype=torch.float)
				next_weight = prox_optimizer.optimize(torch.reshape(prox_input,(1,-1)), 0,hp,prox,regularizer)
				weight_iterates = torch.cat([weight_iterates, torch.reshape(next_weight,(1,-1))])
			
			w_bar = torch.mean(weight_iterates, dim = 0)
			self.stats.compute(weight_iterates[k], loss.compute(X,y,weight_iterates[k]))
		print (self.stats.objective_gap)
		return w_bar
