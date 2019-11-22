import torch


class mse():
    def compute(X, y, w):
        return torch.mean(torch.sub(torch.sum(torch.mul(X, w), dim=1), y)**2)

    def grad(X, y, w):
        return torch.mean(torch.mul(torch.mul(2, torch.sub(torch.sum(
                          torch.mul(X, w), dim=1), y)), X.t()))


class log_loss():
    def compute(X, y, w):
        return torch.mean(torch.log(1. + torch.exp(torch.mul(y,
                          torch.sum(torch.mul(X, w), dim=1)))))

    def grad(X, y, w):
        num = -torch.mul(X.t(), y)
        denom = (1 + torch.exp(y * torch.sum(torch.mul(X, w))))
        return torch.mean(num / denom, dim=1)


class prox_loss():
    def compute(X, y, w):
        return torch.dist(w, X)**2

    def grad(X, y, w):
        return torch.mul(torch.dist(w, X), 2)


class l1_regularizer():
    def compute(w, coeff):
        return torch.mul(torch.dist(w, torch.zeros(w.size()), p=1), 0.5*coeff['l1'])

    def grad(w, coeff):
        if torch.dist(w, torch.zeros(w.size()), p=1) >= 0:
            return torch.mul(torch.ones(w.size()), 0.5*coeff['l1'])
        else:
            return torch.mul(torch.ones(w.size()), -0.5*coeff['l1'])


class l2_regularizer():
    def compute(w, coeff):
        return torch.mul(torch.dist(w, torch.zeros(w.size(), p=2), coeff['l2']))

    def grad(w, coeff):
        return torch.mul(w, 2*coeff['l2'])


class elastic_net_regularizer():
    def compute(w, coeff):
        return torch.add(l1_regularizer.compute(w, coeff),
                         l2_regularizer.compute(w, coeff))

    def grad(w, coeff):
        return torch.add(l1_regularizer.grad(w, coeff),
                         l2_regularizer.grad(w, coeff))
