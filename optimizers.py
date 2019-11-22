from copy import deepcopy
from stats import Stats

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
				grad += regularizer.grad(X,y,w)
			w = w - hp['lr'] * grad
			self.stats.compute(w, loss.compute(X,y,w))
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
		eta, m, num_stages = hp['eta'], hp['m'], hp['s']
		prox_optimizer = Optimizer()
		w_bar = torch.randn(num_params) * 0.01
		for i in range(s):
			v_bar = loss.grad(X, y, wBar)
			weight_iterates = deepcopy(wBar)
			for k in range(1,m + 1):
				q = random.randint(0,numExamples-1)
				v_k = torch.div((loss.grad(X[k],y[k],weight_iterates[k-1]) - loss.grad(X[k],y[k],wBar)),n) + vBar
				prox_input = torch.sub(weight_iterates[k-1], torch.mul(v_k, eta))
				prox_optimizer.optimize(\)
				weight_iterates = torch.stack((weight_iterates, next_weight))
				
			wBar = torch.mean(weight_iterates, dim = 1)

