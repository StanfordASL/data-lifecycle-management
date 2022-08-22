import torch
import numpy as np
from torch import distributions

"""
this file implements different output distributional families, with their loss functions and Fisher matrices

specifically, for each family, if F(theta) = LL^T
we implement left multiplication by L^T using the function

apply_sqrt_F

we also implement:

marginalize(var): which returns the distribution if the parameters used to construct the original 
    distribution were normally distributed with diagonal variance given by var

merge_batch(): returns the distribution approximating the mixture distribution constructed by 
    summing across the first batch dimension. useful if you construct this distribution with a batch
    of parameters corresponding to MC samples, and you want a single distribution approximation of the mixture.

metric(label): which returns a more human friendly measure of error between the distribution and the label
    for Normal distributions, this is the MSE, while for discrete distributions, this yields the 0-1 error

"""

class ExtendedDistribution:
    """
    AbstractBaseClass
    """
    pass

class Bernoulli(distributions.Bernoulli, ExtendedDistribution):
    def __init__(self, probs=None, logits=None, validate_args=None):
        self.use_logits = False
        if probs is None:
            self.use_logits = True
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)
    
    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T, 
        return L^T vec, blocking gradients through L

        if self._param is probs, then 
            F = 1/(p(1-p))
        if self._param is logits, then
            F = p(1-p), where p = sigmoid(logit)
        """
        p = self.probs.detach()
        L = torch.sqrt( p*(1-p) ) + 1e-10 # for stability
        
        if self.use_logits:
            return L * vec
        else:
            return vec / L

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with mean and a diagonal variance as given

        inputs:
            diag_var: variance of parameter (1,)
        """
        if self.use_logits:
            kappa = 1. / torch.sqrt(1. + np.pi / 8 * diag_var)
            p = torch.sigmoid(kappa * self.logits)
        else:
            p = self.probs # gaussian posterior in probability space is not useful
        return Bernoulli(probs=p)

    def merge_batch(self):
        p_mean = self.probs.mean(dim=0)
        return Bernoulli(probs=p_mean)

    def metric(self, y):
        """
        classification error (1- accuracy)
        """
        return ((self.probs >= 0.5) != y).float()

class Normal(distributions.normal.Normal, ExtendedDistribution):
    def __init__(self, loc, scale, validate_args=None):
        super().__init__(loc, scale, validate_args=validate_args)

    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T, 
        return L^T vec, blocking gradients through L

        Here, we assume only loc varies, and do not consider cov as a backpropable parameter
        
        F = Sigma^{-1}

        """
        return vec / self.stddev.detach()

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with a diagonal variance as given

        inputs:
            diag_var: variance of parameter (d,)
        """
        stdev = torch.sqrt(self.variance + diag_var)
        return Normal(loc=self.mean, scale=stdev)

    def merge_batch(self):
        diag_var = torch.mean(self.mean**2, dim=0) - self.mean.mean(dim=0)**2 + self.variance.mean(dim=0)
        return Normal(loc=self.mean.mean(dim=0), scale=torch.sqrt(diag_var))

    def metric(self, y):
        return torch.mean( torch.sum( (y - self.mean)**2, dim=-1) )

class Categorical(distributions.categorical.Categorical, ExtendedDistribution):
    def __init__(self, probs=None, logits=None, validate_args=None):
        self.use_logits = False
        if probs is None:
            self.use_logits = True
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T, 
        return L^T vec, blocking gradients through L

        if self._param is probs, then 
            F = (diag(p^{-1}))
        if self._param is logits, then
            F = (diag(p) - pp^T) = LL^T
        """
        p = self.probs.detach()
        if self.use_logits:
            vec_bar = torch.sum(p*vec, dim=-1, keepdim=True)
            return torch.sqrt(p)*(vec - vec_bar)
        else:
            return vec / (torch.sqrt(p) + 1e-8)

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with a diagonal variance as given

        inputs:
            diag_var: variance of parameter (d,)
        """
        if self.use_logits:
            # @TODO: allow selecting this via an argument
            # probit approx
            kappa = 1. / torch.sqrt(1. + np.pi / 8 * diag_var)
            scaled_logits = kappa*self.logits
            dist = Categorical(logits=scaled_logits)

            # laplace bridge
            # d = diag_var.shape[-1]
            # sum_exp = torch.sum(torch.exp(-self.logits), dim=-1, keepdim=True)
            # alpha = 1. / diag_var * (1 - 2./d + torch.exp(self.logits)/(d**2) * sum_exp)
            # dist = distributions.Dirichlet(alpha)
            # return distributions.Categorical(probs=torch.nan_to_num(dist.mean, nan=1.0))
        else:
            p = self.probs # gaussian posterior in probability space is not useful
            return Categorical(probs=p)
        return dist

    def merge_batch(self):
        p_mean = self.probs.mean(dim=0)
        return Categorical(probs=p_mean)

    def metric(self, y):
        """
        classification error (1- accuracy)
        """
        return (torch.argmax(self.probs, dim=-1) != y).float()