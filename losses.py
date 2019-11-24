import torch

class mse():
    def compute(X, y, w):
        return torch.mean(torch.sub(torch.sum(torch.mul(X, w), dim=1), y.double())**2).double()

    def grad(X, y, w):
        return torch.mean(torch.mul(torch.mul(2., torch.sub(torch.sum(
                          torch.mul(X, w), dim=1), y.double())), X.t()), dim=1)


class log_loss():
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
    def compute(X, y, w):
        return (y.double()-w.double())**2

    def grad(X, y, w):
        return 2*(w.double()-y.double())


class l1_regularizer():
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.mul(torch.norm(w.double(), p=1), coeff['l1']).to(device)

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.norm(w.double(), p=1) >= 0.0:
            return torch.mul(torch.ones(w.size()).to(device),
                             coeff['l1']).to(device)
        else:
            return torch.mul(torch.ones(w.size()).to(device),
                             -coeff['l1']).to(device)


class l2_regularizer():
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.mul(torch.norm(w.to(device), p=2).to(device).double(), 0.5*coeff['l2'])

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.tensor(coeff['l2']*w).double().to(device)


class elastic_net_regularizer():
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.add(l1_regularizer.compute(w.to(device), coeff).double(),
                         l2_regularizer.compute(w.to(device), coeff).double())

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.add(l1_regularizer.grad(w.to(device), coeff).double(),
                         l2_regularizer.grad(w.to(device), coeff).double())


class loss_plus_regulalizer():
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
