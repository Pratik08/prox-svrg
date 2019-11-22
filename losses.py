import torch


class mse():
    def compute(X, y, w):
        return torch.mean(torch.sub(torch.sum(torch.mul(X, w), dim=1), y)**2)

    def grad(X, y, w):
        return torch.mean(torch.mul(torch.mul(2., torch.sub(torch.sum(
                          torch.mul(X, w), dim=1), y)), X.t()))


class log_loss():
    def compute(X, y, w):
        return torch.mean(torch.log(1. + torch.exp(torch.mul(y,
                          torch.sum(torch.mul(X, w), dim=1)))))

    def grad(X, y, w):
        return torch.mean(torch.div(-torch.mul(X.t(), y.double()),
                          (1. + torch.exp(y.double() * torch.sum(torch.mul(X,
                           w.double()))))), dim=1)


class prox_loss():
    def compute(X, y, w):
        return torch.dist(w, X.float())**2

    def grad(X, y, w):
        return torch.mul(torch.dist(w, X.float()), 2)


class l1_regularizer():
    def compute(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.mul(torch.dist(w, torch.zeros(w.size()).to(device), p=1),
                         0.5*coeff['l1']).to(device)

    def grad(w, coeff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.dist(w, torch.zeros(w.size().to(device)), p=1) >= 0:
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
