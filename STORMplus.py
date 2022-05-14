import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required


class STORMplus(Optimizer):

    '''
    >>> Stochastic Recursive Momentum algorithm by [Cutkosy & Orabona, 2019].
    '''

    def __init__(self, params, lr = required, weight_decay = 0., init_accumulator = 1., c = 1.):

        if lr is not required and lr < 0.0:
            raise ValueError('Invalid learning rate: %1.1e'%lr)
        if weight_decay < 0.0:
            raise ValueError('Invalid weight decay value: %1.1e'%weight_decay)

        defaults = dict(lr = lr, weight_decay = weight_decay)

        super(STORMplus, self).__init__(params, defaults)

        self.t = 0
        self.c = c
        self.G_t = 0.
        self.D_t = 0.
        self.estimator_accumulator = init_accumulator
        self.eta_t = 1. / self.estimator_accumulator**(1./3.)
        self.grad_accumulator = init_accumulator
        self.a_t = self.c / self.grad_accumulator**(2./3.)

        # Store the latest estimator for computing the next d_t
        # Store previous state for the gradient computation in the correction step
        # Compute gradient at previous iterate with current batch
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['d_t'] = torch.full_like(p.data, 0.)
                state['current_grad'] = torch.full_like(p.data, 0.)
                state['correction_grad'] = torch.full_like(p.data, 0.)

    def __setstate__(self, state):

        super(STORMplus, self).__setstate__(state)


    '''
    To be called AFTER optimizer.step(), following the SECOND forward/backward pass
    Computes gradient at the next state using next data batch.
    Retrieves current estimator and latest correction gradient.
    Computes next estimator 
    '''
    def compute_estimator(self, normalized_norm = False, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        G_t = 0.
        D_t = 0.
        dimension_normalizer = 0.

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                
                if p.grad is None:
                    continue
                p_grad = p.grad.data # next gradient
                if weight_decay != 0:
                    p_grad.add_(weight_decay, p.data)

                state = self.state[p]
                state['current_grad'] = p_grad.detach()
                state['d_t'] = ( state['current_grad'] + ( 1. - self.a_t ) * ( state['d_t'] - state['correction_grad'] ) ).detach()

                G_t += torch.linalg.norm(p_grad)**2
                D_t += torch.linalg.norm(state['d_t'])**2
                if normalized_norm:
                    dimension_normalizer += np.prod(p.shape)
        
        self.G_t = G_t / dimension_normalizer if normalized_norm else G_t
        self.D_t = D_t / dimension_normalizer if normalized_norm else D_t
    

    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        # Compute momentum parameter, which requires the previous step-size


        # Compute the denominator of the step-size by going over all param groups
        # Since we need the norm-squared, we could safely add norm-squared of gradient wrt each param group
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                p_grad = p.grad.data # next correction grad
                if weight_decay != 0:
                    p_grad.add_(weight_decay, p.data)
                
                state = self.state[p]
                state['correction_grad'] = p_grad.detach()

        
        self.grad_accumulator += self.G_t
        self.a_t = self.c / self.grad_accumulator**(2./3.)
        self.estimator_accumulator += self.D_t / self.a_t
        self.eta_t = 1. / self.estimator_accumulator**(1./3.)

        for group in self.param_groups:
            lr = group['lr']
            eta_t = lr * self.eta_t
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                p_grad = p.grad.data
                if weight_decay != 0:
                    p_grad.add_(weight_decay, p.data)
                
                state = self.state[p]

                p.data.add_(-eta_t, state['d_t'])
                

        return loss

    def lr(self):

        lr_list = []
        for group in self.param_groups:
            lr = group['lr']
            lr_list.append(float(lr * float(self.eta_t)))

        out = float( sum(lr_list) / len(lr_list) )

        return out
