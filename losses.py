'''
Authors: Pratik Dubal, Shashwat Verma and Saurabh Sharma.

All the code is vectorized and cuda enabled.
'''

import torch


class mse():
    '''
    Computes the mean squared error and its gradient.
    MSE is defined as (a-b)^2
    '''
    def compute(X, y, w):
        return torch.mean(torch.sub(torch.sum(torch.mul(X, w), dim=1), y.double())**2).double()

    def grad(X, y, w):
        return torch.mean(torch.mul(torch.mul(2., torch.sub(torch.sum(
                          torch.mul(X, w), dim=1), y.double())), X.t()), dim=1)


class log_loss():
    '''
    Computes the logistic loss and its gradient.
    Defined as log(1 + e(-ywx))
    '''
    def compute(X, y, w):
        return torch.mean(torch.log(1. + torch.exp(-1. * torch.mul(y, torch.sum(torch.mul(X, w), dim=1))))).double()

    def grad(X, y, w):
        y_hat = torch.sum(torch.mul(X, w), dim=1).double()
        t1 = torch.div(torch.exp(-1. * torch.mul(y, y_hat)),
                       1. + torch.exp(-1. * torch.mul(y, y_hat))).double()
        t2 = torch.mul(-1. * y.repeat(1, X.size(1)).reshape(X.size(1),
                       X.size(0)), X.t()).double()
        w_ret = torch.div(torch.mul(t2, t1).sum(dim=1), X.size(0)).double()
        return w_ret


class prox_loss():
    '''
    Computes the value for the proximal mapping used in the
    experiments, along with its gradient.
    '''
    def compute(X, y, w):
        return 0.5*torch.norm(w.double() - y.double(), p=2)**2

    def grad(X, y, w):
        return (w.double()-y.double())


class l1_regularizer():
    '''
    Computes the l1 regularization value for the weights
    and its sub-gradient.
    '''
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.mul(torch.norm(w.double(), p=1), coeff['l1']).to(device)

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = (w > 0.0).int()
        return (coeff['l1'] * mask - coeff['l1'] * (1 - mask)).double().to(device)


class l2_regularizer():
    '''
    Computes the l2 regularization value for the weights
    and its gradient.
    '''
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.mul((torch.norm(w.to(device), p=2)**2).to(device).double(), 0.5*coeff['l2'])

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (coeff['l2']*w).double().to(device)


class elastic_net_regularizer():
    '''
    Computes the elastic regularization and its gradient
    '''
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.add(l1_regularizer.compute(w.to(device), coeff).double(),
                         l2_regularizer.compute(w.to(device), coeff).double())

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.add(l1_regularizer.grad(w.to(device), coeff).double(),
                         l2_regularizer.grad(w.to(device), coeff).double())


class loss_plus_regulalizer():
    '''
    Computes the value for the objective function
    (log_loss and elastic regularization in the experiments).
    Different losses and regularizations can also be used.
    '''
    def compute(X, y, w, coeff, loss, regularizer):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if regularizer is not None:
            return loss.compute(X, y, w).to(device) + regularizer.compute(w, coeff).to(device)
        return loss.compute(X, y, w)

    def grad(X, y, w, coeff, loss, regularizer):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if regularizer is not None:
            return loss.grad(X, y, w).to(device) + regularizer.grad(w, coeff).to(device)
        return loss.grad(X, y, w).to(device)
