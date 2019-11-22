import os
import tqdm
import torch
import numpy as np


class Dataset:
    def __init__(self, X, y, verbose=True):
        self.X = X
        self.y = y

        if verbose:
            print("Number of rows in X: {}".format(self.X.size(0)))
            print("Number of dimensions in X: {}".format(self.X.size(1)))
            print("Number of rows in y: {}".format(self.y.size(0)))

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


class Rcv1(Dataset):
    def __init__(self, path='../data/rcv1', verbose=True):
        if verbose:
            print("Loading Rcv1 dataset...")

        # initalize inputs and targets
        X = []
        y = []
        # datapoints are $dim$ dimensional
        dim = 47236

        X_path = os.path.join(path, 'rcv1_train.binary')
        with open(X_path, 'r') as f:
            lines = f.readlines()
            pbar = tqdm.tqdm(total=len(lines))
            for line in lines:
                x_feat = line.split(' ')
                x = np.zeros(dim)
                for feat in x_feat[1:]:
                    x[int(feat.split(':')[0])-1] = float(feat.split(':')[1])
                X.append(x)
                y.append(int(x_feat[0]))
                pbar.update(1)
            pbar.close()
        X = torch.from_numpy(np.asarray(X))
        y = torch.from_numpy(np.asarray(y))
        super().__init__(X, y, verbose)


class Sido(Dataset):
    def __init__(self, path='../data/sido0', verbose=True):
        if verbose:
            print("Loading Sido0 dataset...")

        # get input data
        X_path = os.path.join(path, 'sido0_train.data')
        X = []
        with open(X_path, 'r') as f:
            lines = f.readlines()
            pbar = tqdm.tqdm(total=len(lines))
            for line in lines:
                x = np.asarray([float(int(x)) for x in line.strip().split(" ")])
                X.append(x)
                pbar.update(1)
            pbar.close()
        X = torch.from_numpy(np.asarray(X))

        # get labels
        y_path = os.path.join(path, 'sido0_train.targets')
        y = []
        with open(y_path, 'r') as f:
            lines = f.readlines()
            pbar = tqdm.tqdm(total=len(lines))
            for line in lines:
                if line == ' ':
                    continue
                y.append(int(line.strip()))
                pbar.update(1)
            pbar.close()
        y = torch.from_numpy(np.asarray(y))
        super().__init__(X, y, verbose)