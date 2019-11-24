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
                          torch.sum(torch.mul(X, w), dim=1)))))

    def grad(X, y, w):
        y_hat = torch.sum(torch.mul(X, w), axis=1)
        t1 = torch.div(torch.exp(-1. * torch.mul(y, y_hat)), 1. + torch.exp(-1. * torch.mul(y, y_hat)))
        t2 = torch.mul(-1. * y.repeat(1, X.size(1)).reshape(X.size(1), y.size(0)), X.t())
        w_ret = torch.div(torch.mul(t2, t1).sum(axis=1), X.size(0))
        return w_ret


class prox_loss():
    def compute(X, y, w):
        return 0.5 * torch.dist(w, X.float())**2

    def grad(X, y, w):
        return torch.dist(w, X.float())


class l1_regularizer():
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.mul(torch.dist(w, torch.zeros(w.size()).to(device), p=1),
                         0.5*coeff['l1']).to(device)

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.dist(w, torch.zeros(w.size()).to(device), p=1) >= 0:
            return torch.mul(torch.ones(w.size()).to(device),
                             0.5*coeff['l1']).to(device)
        else:
            return torch.mul(torch.ones(w.size()).to(device),
                             -0.5*coeff['l1']).to(device)


class l2_regularizer():
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.mul(torch.dist(w, torch.zeros(w.size(), p=2).to(device),
                                    coeff['l2']))

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.mul(w, 2*coeff['l2']).to(device)


class elastic_net_regularizer():
    def compute(w, coeff):
        return torch.add(l1_regularizer.compute(w, coeff),
                         l2_regularizer.compute(w, coeff))

    def grad(w, coeff):
        return torch.add(l1_regularizer.grad(w, coeff),
                         l2_regularizer.grad(w, coeff))
