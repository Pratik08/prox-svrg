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
        return torch.mean()


class prox_loss():
    def compute(C, w, regularizer):
        return torch.sum(torch.pow(torch.dist(w,C),2),regularizer.compute(w))

    def grad(C, w, regularizer):
        return torch.mean(torch.sum(torch.mul(torch.sub(w,C),2),regularizer.grad))

class l1_regularizer():
    def compute(w, coeff):
        return torch.mul(torch.dist(w,torch.zeros(w.size(),p=1), 0.5*coeff))

    def grad(w, coeff):
        if torch.dist(w,torch.zeros(w.size(),p=1)) >= 0:
            return torch.mul(torch.ones(w.size()), 0.5*coeff)
        else:
            return torch.mul(torch.ones(w.size()), -0.5*coeff)

class l2_regularizer():
    def compute(w, coeff):
        return torch.mul(torch.dist(w,torch.zeros(w.size(),p=2), coeff))

    def grad(w, coeff):
        return torch.mul(w, 2*coeff)
