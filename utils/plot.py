import matplotlib.pyplot as plt


def plot_num_non_zeros(stats):

    nnzs = []
    effective_passes = []
    og = []

    for stat in stats:
        nnz = stat.num_non_zeros
        nnzs.append(nnz)

        ep = stat.effective_passes
        eps.append(ep)

        og = stat.objective_gaps
        ogs.append(og)

    # plot nnzs

    # plot eps

    # plot ogs
