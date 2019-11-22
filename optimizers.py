'''
	Base class with a Gradient Descent optimizer
	Init params:
		data ???Dataframe???
		hp - hyperparameters
		objectiveFunction - Function we are trying to minimze
		regularizer - Regularization function
		objectiveGradient - Function which returns gradient of objective function
		regularizerGradient - Function returns regularizer Gradient

'''
class Optimizer:
	def __init__(self, data, hp, objectiveFunction, regularizer, objectiveGradient, regularizerGradient):
		self.data = data
		self.X = data['X']
		self.y = data['y']
		self.statistics = Statistics()
		self.learningRate = hp['learning_rate']
		self.objectiveFunction = objectiveFunction
		self.objectiveGradient = gradient
		self.regularizer = regularizer
		self.regularizerGradient = regularizerGradient


	def optimize(self):
		utils.GD()