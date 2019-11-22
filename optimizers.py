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

	def optimize(self, X, y, hp, loss, regularizer, prox = None):
		# Gradient Descent
		return None



'''
	Hyperparamters: 
		s - number of stages
		m - number of iterations per stage

'''
class ProxSVRGOptimizer(Optimizer):
	def __init__(self, data, hp, objectiveFunction, regularizer, objectiveGradient, regularizerGradient):
		super().__init__(data, hp, objectiveFunction, regularizer, objectiveGradient, regularizerGradient)
		# 

	def run_stage(self, ):

	def optimize(self, X, y, hp, loss, regularizer, prox):
		numParams = X.size(1)
		numExamples = X.size(0)
		eta = hp['eta']
		m = hp['m']
		epsilon = 
		weights = torch.from_numpy(np.zeros(numParams))
		wBar = deepcopy(weights)
		for i in range(self.hp['stages']):
			loss.compute(X,y,wBar)
			w_0 = deepcopy(weights)
			q = 
			self.run_stage()

