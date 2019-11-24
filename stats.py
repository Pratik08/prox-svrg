import torch
import matplotlib.pyplot as plt


class Stats:
    """
    This class is used for computing statistics
    such as number on non zeros, objective gap, 
    and effective passes. Only compute method 
    is the public method. 
    """
    def __init__(self):
        self.num_non_zeros = []
        self.objective_gap = []
        self.effective_passes = []
        self.current_iteration = 0

    def _compute_num_non_zeros(self, w):
        self.num_non_zeros.append(int((torch.abs(w) > 1e-5).int().sum().data.cpu().numpy()))

    def _compute_objective_gap(self, loss):
        self.objective_gap.append(loss)

    def _compute_effective_passes(self):
        self.effective_passes.append(self.current_iteration)

    def compute(self, w, loss):
        """
        w: tensor of weights [1 x d]
        loss: float
        """
        self._compute_num_non_zeros(w)
        self._compute_objective_gap(loss)
        self._compute_effective_passes()
        self.current_iteration += 1

    def plot(self,title = ""):
        plt.plot(self.num_non_zeros)
        plt.xlabel("Effective Passes")
        plt.ylabel("NNZs")
        plt.title(title+" NNZs")
        plt.xticks(range(len(self.num_non_zeros)))
        plt.savefig(title+"_NNZs.png")
        plt.clf()
        plt.plot(self.objective_gap)
        plt.xlabel("Effective Passes")
        plt.ylabel("Objective Loss")
        plt.title(title+" Loss")
        plt.xticks(range(len(self.num_non_zeros)))
        plt.savefig(title+"_Loss.png")
        plt.clf()