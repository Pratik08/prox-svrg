class Stats:

    def __init__(self):
        self.num_non_zeros = []
        self.objective_gap = []
        self.effective_passes = []
        self.current_iteration = 0

    def _compute_num_non_zeros(self, w):
        self.num_non_zeros.append(int((w != 0).int().sum().data.cpu().numpy()))

    def _compute_objective_gap(self, loss):
        self.objective_gap.append(loss)

    def _compute_effective_passes(self):
        self.effective_passes.append(self.current_iteration)

    def compute(self, w, loss):
        self._compute_num_non_zeros(w)
        self._compute_objective_gap(loss)
        self._compute_effective_passes()
        self.current_iteration += 1