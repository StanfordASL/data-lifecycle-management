from typing import Union, Optional, Callable, Tuple, List
import torch
from torch import nn
from copy import deepcopy
from torch.cuda.amp.autocast_mode import autocast

from tqdm import tqdm
from torch.autograd import grad
from .sketching.sketched_pca import alg_registry
from .distributions import ExtendedDistribution

import numpy as np

class Projector(nn.Module):
    def __init__(self, N : int, 
                 r : int, 
                 device : torch.DeviceObjType = torch.device('cpu')) -> None:
        """
        Class which handles computations comparing a test-time jacobian to
        the top eigenvalues and eigenvectors of the Gauss Newton matrix
        Inputs: N, r, device
            expects eigenvectors of dim N x r
            eigenvalues of dim r
        """
        super().__init__()
        self.N = N
        self.r = r
        
        self.device = device
        torch.manual_seed(11)
        self.eigs = nn.Parameter(1e-8*torch.ones(self.r, device=self.device), requires_grad=False)
        self.basis = nn.Parameter(torch.randn(self.N, self.r, device=self.device), requires_grad=False)
        
    @torch.no_grad()
    def process_basis(self, eigs : torch.Tensor,
                      basis : torch.Tensor) -> None:
        n_eigs = eigs.shape[0]
        self.eigs.data[-n_eigs:] = eigs.to(self.device)
        self.basis.data[:,-n_eigs:] = basis.to(self.device)

    def ortho_proj(self, L : torch.Tensor, 
                   n_eigs : torch.Tensor) -> torch.Tensor:
        """
        we have U = basis,
        computes ||(I-UU^T)L||^2_F
        """
        basis = self.basis[:,-n_eigs:]
        proj_L = basis.t() @ L 
        proj_L = basis @ proj_L
        return torch.norm(L - proj_L) 
    
    def posterior_pred(self, L : torch.Tensor, 
                       n_eigs : torch.Tensor, 
                       eps : torch.Tensor) -> torch.Tensor:
        """
        we have U = basis,
        computes ||(I-UU^T)L||^2_F
        """
        basis = self.basis[:,-n_eigs:]
        eigs = torch.clamp( self.eigs[-n_eigs:], min=0.)

        scaling = torch.sqrt( eigs / ( eigs + 1./(eps) ) )
        proj_L = scaling[:,None] * (basis.t() @ L)

        return torch.sqrt( torch.sum(L**2) - torch.sum(proj_L**2) )
    
    @torch.no_grad()
    def compute_distance(self, L : torch.Tensor, 
                         proj_type : torch.Tensor, 
                         n_eigs : Optional[int] = None, 
                         eps : Union[torch.Tensor , float] =1.) -> torch.Tensor:
        if n_eigs is None:
            n_eigs = self.r
            
        L.to(self.device)
        if proj_type == 'ortho':
            return self.ortho_proj(L, n_eigs)
        elif proj_type == 'posterior_pred':
            return self.posterior_pred(L, n_eigs, eps)
        else:
            raise ValueError(proj_type +" is not an understood projection type.")

    def compute_diag_var(self, J : torch.Tensor, 
                               P : Optional[torch.Tensor] = None, 
                               n_eigs : Optional[int] = None, 
                               eps : Union[torch.Tensor, float] = 1.) -> torch.Tensor:
        """
        given J, (d, N)
        
        if P is not given, assumes P is the identity matrix

        returns diagonal variance of P J Sig J^T P^T, where
            Sig = ( 1/eps I + M U D U^T )^{-1}
                = eps I - eps U (1/(eps) D^{-1} + I)^{-1} U^T
        """
        basis = self.basis[:,-n_eigs:]
        eigs = torch.clamp( self.eigs[-n_eigs:], min=0.)
        

        JJT = J @ J.T # torch.sum(J**2, dim=-1)

        scaling = eigs / (eigs + 1./ (eps))
        neg_term = torch.sqrt(scaling[None,:])*(J @ basis)
        neg_term = neg_term @ neg_term.T
        
        if P is not None:
            chol_sig = torch.linalg.cholesky(JJT - neg_term)
            sig = torch.sum( (P @ chol_sig)**2, dim=-1)  #( P[:,None,:] @ ( (JJT - neg_term) @ P.T) )[:,0]
            return eps*sig
        
        else: 
            return torch.diagonal(eps*(JJT - neg_term)) # shape (d,d)

base_config = {
    'num_samples': None, # sketch size T (T)
    'num_eigs': 10, # low rank estimate to recover (k)
    'sketch_type': 'gaussian', # sketch type 
}


class SCOD(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    """
    def __init__(self, model : nn.Module, 
                       dist_constructor : Callable[[torch.Tensor], ExtendedDistribution], 
                       args : dict = {}, 
                       parameters : Optional[nn.ParameterList] = None) -> None:
        """
        model: base DNN to equip with an uncertainty metric
        dist_constructor: a function mapping network output to a Distribution object, defining a distribution
            over the output space. The labels in a dataset should lie in the support of this distribution.
        args: configuration variables - defaults are in base_config
            'num_samples': default=None, otherwise int > 0, sketch dimension T (T)
            'num_eigs': default=10, otherwise int > 0 low rank estimate to recover (k)
            'sketch_type': default='srft', sketch type either 'gaussian' or 'srft' for
                linear sketching techniques, or 'ipca' for an incremental PCA approach (much slower)
        """
        super().__init__()
        
        self.config = deepcopy(base_config)
        self.config.update(args)
        
        self.model = model
        self.dist_constructor = dist_constructor

        # extract device from model
        self.device = next(model.parameters()).device
        
        # extract parameters to consider in sketch - keep all that yield valid gradients
        if parameters is None:
            self.trainable_params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        else:
            self.trainable_params = list(parameters)
        self.n_params = int(sum(p.numel() for p in self.trainable_params))
                
        self.num_samples = self.config['num_samples']
        self.num_eigs = self.config['num_eigs']
        
        if self.num_samples is None:
            self.num_samples = 6*self.num_eigs + 4
            
        self.configured = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        
        self.sketch_class = alg_registry[self.config['sketch_type']] 
        print(self.sketch_class)
        
        self.projector = Projector(
            N=self.n_params,
            r=2*max(self.num_eigs + 2, (self.num_samples-1)//3),
            device=self.device
        )

        self.log_eps = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True) # prior variance
        self.scaling_factor = nn.Parameter(torch.ones(1, device=self.device), requires_grad=False) # final scaling factor to rescale uncertainty output
        self.hyperparameters = [self.log_eps]

    @property
    def eps(self) -> torch.Tensor:
        """
        returns eps, the scale on the prior covariance
        """
        return torch.exp(self.log_eps)

    def process_dataloader(self, dataloader : torch.utils.data.DataLoader ) -> None:
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are 
        taken to be irrelevant to data, and used for detecting generalization
        
        dataloader - torch dataloader of (input, target) pairs
        """
        # loop through data, one sample at a time
        print("computing basis") 
        
        sketch = self.sketch_class(N=self.n_params, 
                                   r=self.num_eigs,
                                   T=self.num_samples,
                                   device=self.device)


        # n_data = len(dataloader)
        n_data = self.num_samples
        for i, (inputs,labels,_) in tqdm(enumerate(dataloader), total=n_data):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast():
                z = self.model(inputs) # get params of output dist
            
                dist = self.dist_constructor(z)
                Lt_z = dist.apply_sqrt_F(z).mean(dim=0) # L^\T theta
                # flatten
                Lt_z = Lt_z.view(-1)

            
            Lt_J = self._get_weight_jacobian(Lt_z) # L^\T J, J = dth / dw
            sketch.low_rank_update(Lt_J.t()) # add J^T L L^T J to the sketch
        
            del Lt_J
        eigs, eigvs = sketch.eigs()
        del sketch

        self.projector.process_basis(eigs, eigvs)
            
        self.configured.data = torch.ones(1, dtype=torch.bool)

    def process_dataset(self, dataset : torch.utils.data.Dataset ) -> None:
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are 
        taken to be irrelevant to data, and used for detecting generalization
        
        dataset - torch dataset of (input, target) pairs
        """
        # loop through data, one sample at a time
        print("computing basis") 
            
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=1, 
                                                 shuffle=True,
                                                 num_workers=0,
                                                 pin_memory=True)
        
        sketch = self.sketch_class(N=self.n_params, 
                                   r=self.num_eigs,
                                   T=self.num_samples,
                                   device=self.device)


        n_data = len(dataloader)
        for i, (inputs,labels) in tqdm(enumerate(dataloader), total=n_data):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast():
                z = self.model(inputs) # get params of output dist
            
                dist = self.dist_constructor(z)
                Lt_z = dist.apply_sqrt_F(z).mean(dim=0) # L^\T theta
                # flatten
                Lt_z = Lt_z.view(-1)

            
            Lt_J = self._get_weight_jacobian(Lt_z) # L^\T J, J = dth / dw
            sketch.low_rank_update(Lt_J.t()) # add J^T L L^T J to the sketch
        
        del Lt_J
        eigs, eigvs = sketch.eigs()
        del sketch

        self.projector.process_basis(eigs, eigvs)
            
        self.configured.data = torch.ones(1, dtype=torch.bool)

    
    def _get_weight_jacobian(self, vec : torch.Tensor) -> torch.Tensor:
        """
        returns k x nparam matrix, with each row being d(vec[i])/d(weights) for i = 1, ..., k
        """
        assert len(vec.shape) == 1
        grad_vecs = []
        for j in range(vec.shape[0]):
            grads = grad(vec[j], self.trainable_params, retain_graph=True, only_inputs=True, allow_unused=True)
            g = torch.cat([gr.contiguous().view(-1) for gr in grads]).detach()
            grad_vecs.append(g)
            
        return torch.stack(grad_vecs)
    
    def get_J_z_s(self, inputs : torch.Tensor) -> List[torch.Tensor] :
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
                            
        returns 
            J_z_s = list of N weight Jacobians
        """
        if not self.configured:
            print("Must call process_dataset first before using model for predictions.")
            raise NotImplementedError
        
            
        N = inputs.shape[0]
        
        z = self.model(inputs)        

        J_z_s = []

        for j in range(N):
            J_z = self._get_weight_jacobian(z[j,...].view(-1)).detach()
            J_z_s.append( J_z )

        return J_z_s
            
    def forward(self, inputs : torch.Tensor, 
                      n_eigs : Optional[int] = None, 
                      proj_type : str = "posterior_pred") -> Tuple[ List[torch.distributions.Distribution], torch.Tensor ] :
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
                            
        returns 
            mu = model(inputs) list of N distribution objects
            unc = hessian based uncertainty estimates shape (N), torch.Tensor
        """
        if not self.configured:
            print("Must call process_dataset first before using model for predictions.")
            raise NotImplementedError
                    
        if n_eigs is None:
            n_eigs = self.num_eigs
            
        N = inputs.shape[0]
        
        z = self.model(inputs)        

        unc = []
        dists = []

        for j in range(N):
            J_z = self._get_weight_jacobian(z[j,...].view(-1)).detach()
            dist = self.dist_constructor(z[j,...])

            f_var = self.projector.compute_diag_var(J_z, n_eigs=n_eigs, eps=self.eps)
            f_var = f_var.reshape(z[j,...].shape)

            output_dist = dist.marginalize(f_var)

            dists.append( output_dist )
            unc.append( (output_dist.entropy() / self.scaling_factor ).sum())

        unc = torch.stack(unc)
        return dists, unc

    def calibrate(self, val_dataset, percentile=0.99) -> None:
        """
        evalutes the uncalibrated score on the val_dataset, 
        and then selects a
        scaling factor such that percentile % of the dataset are below 1
        """
        scores = []

        dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=10)

        for i, (inputs,labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs, uncs = self.forward(inputs)

            scores.append(uncs)
        
        scores = torch.cat(scores)
        value = torch.quantile(scores, percentile)

        self.scaling_factor.data *= value

    def calibrate_dataloader(self, val_dataloader, percentile=0.99) -> None:
        """
        evalutes the uncalibrated score on the val_dataloader, 
        and then selects a
        scaling factor such that percentile % of the dataset are below 1
        """
        scores = []

        for i, (inputs,labels,_) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs, uncs = self.forward(inputs)

            scores.append(uncs)
        
        scores = torch.cat(scores)
        value = torch.quantile(scores, percentile)

        self.scaling_factor.data *= value

