import torch


class mse():
    def compute(X, y, w):
        return torch.mean(torch.sub(torch.sum(torch.mul(X, w), dim=1), y)**2)

    def grad(X, y, w):
        return torch.mean(torch.mul(torch.mul(2., torch.sub(torch.sum(
                          torch.mul(X, w), dim=1), y)), X.t()), axis=1)


class log_loss():
    def compute(X, y, w):
        return torch.mean(torch.log(1. + torch.exp(-1. * torch.mul(y,
                          torch.sum(torch.mul(X, w), dim=1))))).double()

    def grad(X, y, w):
        y_hat = torch.sum(torch.mul(X, w), axis=1).double()
        t1 = torch.div(torch.exp(-1. * torch.mul(y, y_hat)),
                       1. + torch.exp(-1. * torch.mul(y, y_hat))).double()
        t2 = torch.mul(-1. * y.repeat(1, X.size(1)).reshape(X.size(1),
                       X.size(0)), X.t()).double()
        w_ret = torch.div(torch.mul(t2, t1).sum(axis=1), X.size(0)).double()
        return w_ret


class prox_loss():
    def compute(X, y, w):
        return (y-w)**2

    def grad(X, y, w):
        return 2*(w-y)


class l1_regularizer():
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.mul(torch.dist(w.double(), torch.zeros(w.size()).double().to(device), p=1),
                         coeff['l1']).to(device)

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.dist(w.double(),
                      torch.zeros(w.size()).double().to(device), p=1) >= 0:
            return torch.mul(torch.ones(w.size()).to(device),
                             coeff['l1']).to(device)
        else:
            return torch.mul(torch.ones(w.size()).to(device),
                             -coeff['l1']).to(device)


class l2_regularizer():
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.mul(torch.dist(w, torch.zeros(w.size()).double(), p=2).to(device),
                                    0.5*coeff['l2'])

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return coeff['l2']*w


class elastic_net_regularizer():
    def compute(w, coeff):
        return torch.add(l1_regularizer.compute(w, coeff),
                         l2_regularizer.compute(w, coeff))

    def grad(w, coeff):
        return torch.add(l1_regularizer.grad(w, coeff),
                         l2_regularizer.grad(w, coeff))


class loss_plus_regulalizer():
    def compute(X, y, w, coeff, loss, regularizer):
        if regularizer is not None:
            return loss.compute(X, y, w) + regularizer.compute(w, coeff)
        return loss.compute(X, y, w)

    def grad(X, y, w, coeff, loss, regularizer):
        if regularizer is not None:
            return loss.grad(X, y, w) + regularizer.grad(w, coeff)
        return loss.grad(X, y, w)
