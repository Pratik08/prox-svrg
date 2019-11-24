import tqdm
import torch
import random
from copy import deepcopy
from stats import Stats
from losses import loss_plus_regulalizer


class Optimizer:
    '''
    Base class with Gradient Descent optimizer
    optimize is a function which will be overloaded by the derived class.
    optimize paramerts:
        X - data
        y - label
        hp - dict of hyperparameters
        loss - object with F part of objective function
        regularizer - object with R part of objective function
        prox - object with proximal mapping function
    '''

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.stats = Stats()

    def prox_mapping_l2(self, hp, w):
        t = hp['coeff']['l2']
        l2_norm = torch.dist(w, torch.zeros(w.size()).double(), p=2).double()
        if l2_norm >= t:
            return torch.mul(w, 1-torch.div(t,l 2_norm).double()).double()
        else:
            return (torch.zeros(w.size())).double()

    def prox_mapping_l1(self, hp, w):
        t = hp['coeff']['l1']
        p = torch.max(torch.abs(w)-t,torch.zeros(w.size()).double())
        if w.sum().data[0] == 0:
        	return (torch.zeros(w.size())).double()
        return p*torch.div(w,abs(w))


    def prox_elastic_net(self, hp, w):
        prox_l1 = self.prox_mapping_l1(hp, w.double())
        return self.prox_mapping_l2(hp, prox_l1.double())

    def optimize(self, X, y, hp, loss, regularizer=None, prox=None,
                 verbose=False):
        # Gradient
        w = 0.01 * torch.randn(X.size(1)).to(self.device).double()

        if verbose:
            pbar = tqdm.tqdm(total=hp['max_iter'])
        for i in range(hp['max_iter']):
            grad = loss.grad(X, y, w)
            if regularizer is not None:
                grad = torch.add(grad, regularizer.grad(w, hp['coeff']))
            w = (w - hp['lr'] * grad).double()
            self.stats.compute(w, loss.compute(X, y, w))
            # print("Stage: %d Loss: %f NNZs: %d" % (i, self.stats.objective_gap[-1],self.stats.num_non_zeros[-1]))
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
        return w


class ProxSVRGOptimizer(Optimizer):
    '''
    Hyperparamters:
    s - number of stages
    m - number of iterations per stage
    eta - learning rate
    '''

    def __init__(self):
        super().__init__()

    def optimize(self, X, y, hp, loss, regularizer, prox, dataset="SIDO"):
        n_examples, n_params = X.size(0), X.size(1)
        eta, m, s = hp['eta'], hp['m'], hp['s']
        prox_optim = Optimizer()

        w_bar = 0.1 * torch.randn(n_params).double().to(self.device)

        pbar = tqdm.tqdm(total=(s*m))
        ctr = 0
        for i in range(s+1):
            v_bar = loss.grad(X, y, w_bar)
            w_itrs = torch.reshape(w_bar, (1, -1))

            for k in range(1, m + 1):
                if (ctr % n_examples == 0):
                    l = loss_plus_regulalizer.compute(X, y, w_bar, hp['coeff'],
                                                      loss, regularizer)
                    self.stats.compute(w_bar, l)
                ctr += 1
                q = torch.randint(n_examples, (1, 1)).item()
                p1 = loss.grad(torch.reshape(X[q], (1, -1)), y[q],
                               w_itrs[k-1])
                p2 = loss.grad(torch.reshape(X[q], (1, -1)), y[q], w_bar)
                v_k = p1 - p2 + v_bar
                prox_input = w_itrs[k-1] - eta * v_k
                nxt_w = self.prox_elastic_net(hp, prox_input)
                # nxt_w = prox_optim.optimize(torch.eye(n_params),
                #                             prox_input, hp, prox, regularizer)
                # nxt_w = prox_input
                w_itrs = torch.cat([w_itrs,
                                    torch.reshape(nxt_w.double(), (1, -1))])
                pbar.update(1)
            w_bar = torch.mean(w_itrs, dim=0)
            print("Stage: %d Loss: %f NNZs: %d" % (i, self.stats.objective_gap[-1],self.stats.num_non_zeros[-1]))

        pbar.close()
        self.stats.plot("ProxSVRG_"+dataset)
        return w_bar


class ProxSAGOptimizer(Optimizer):
	'''
	Hyperparamters:
	m - number of iterations
	eta - learning rate
	'''

	def __init__(self):
		super().__init__()

	def optimize(self, X, y, hp, loss, regularizer, prox,  dataset="SIDO"):
		n_examples, n_params = X.size(0), X.size(1)
		d = torch.zeros(n_params).double().to(self.device)
		prev_grads = torch.zeros((n_examples, n_params)).double().to(self.device)
		prox_optim = Optimizer()
		w = 0.1 * torch.randn(X.size(1)).double().to(self.device)
		ctr = 0
		pbar = tqdm.tqdm(total=(hp['m']))
		for k in range(hp['m'] + 1):
			if (k%n_examples == 0):
				l = loss_plus_regulalizer.compute(X,y,w,hp['coeff'],loss,regularizer)
				self.stats.compute(w, l)
				print("Stage: %d Loss: %f NNZs: %d" % (k/n_examples, self.stats.objective_gap[-1],self.stats.num_non_zeros[-1]))
			q = torch.randint(n_examples, (1, 1)).item() # Replace with sampling function (?)
			x_sample = torch.reshape(X[q],(1,-1)).double().to(self.device)
			y_sample = y[q]
			grad_sample = loss.grad(x_sample,y_sample,w)
			d = torch.add(torch.sub(d,prev_grads[q]),grad_sample)
			prev_grads[q] = grad_sample
			prox_input = torch.sub(w,torch.mul(d,hp['eta']/n_examples).double()).double()
			w = self.prox_elastic_net(hp,prox_input.double()).double()
            # w = prox_optim.optimize(torch.eye(n_params),
            #                                 prox_input, hp, prox, regularizer).double()
			# w = prox_input.clone()
			pbar.update(1)

		pbar.close()
		self.stats.plot("ProxSAG_"+dataset)
		return w


class ProxSGOptimizer(Optimizer):
	'''
	Hyperparamters:
	m - number of iterations
	eta - learning rate
	'''

	def __init__(self):
		super().__init__()

	def optimize(self, X, y, hp, loss, regularizer, prox,  dataset = "SIDO"):
		n_examples, n_params = X.size(0), X.size(1)
		d = torch.zeros(n_params).double().to(self.device)
		prev_grads = torch.zeros((n_examples,n_params)).double().to(self.device)
		prox_optim = Optimizer()
		w = 0.1 * torch.randn(X.size(1)).double().to(self.device)
		pbar = tqdm.tqdm(total=(hp['m']))
		eta = hp['eta']
		for k in range(hp['m']+1):
			if k % (n_examples/100) == 0:
				l = loss_plus_regulalizer.compute(X,y,w,hp['coeff'],loss,regularizer)
				self.stats.compute(w, l)
				print("Stage: %d Loss: %f NNZs: %d" % (k / n_examples, self.stats.objective_gap[-1],self.stats.num_non_zeros[-1]))
				eta /= 2
			q = torch.randint(n_examples, (1, 1)).item() # Replace with sampling function (?)
			x_sample = torch.reshape(X[q],(1,-1)).double().to(self.device)
			y_sample = y[q]
			grad_sample = loss.grad(x_sample,y_sample,w)
			prox_input = torch.sub(w,torch.mul(grad_sample,eta).double()).double()
			w = self.prox_elastic_net(hp,prox_input.double()).double()
            # w = prox_optim.optimize(torch.eye(n_params),
            #                                 prox_input, hp, prox, regularizer)
			# w = prox_input.clone()
			pbar.update(1)
		self.stats.plot("ProxSG_"+dataset)
		pbar.close()
		return w
