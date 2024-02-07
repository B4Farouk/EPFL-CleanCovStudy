from numbers import Complex

import math
import numpy
import numpy as np
import torch

from sklearn.covariance import (
    ledoit_wolf as sklearn_ledoit_wolf,
    oas as sklearn_oas
    )
import bahc

from covariance.estimation import get_covariance_estimator, corr_estimator
from covariance.numerr_cleaning import numerr_clean_cov_mat, numerr_clean_corr_mat
from utils import to_torch_tensor

def get_covariance_cleaner(cleaner: str) -> callable:
    if cleaner == None: #because can't call None.lower()
        return no_cleaning
    match cleaner.lower():
        case None: return no_cleaning
        case "clipping": return clipping
        case "linear_shrinkage": return linear_shrinkage
        case "oas": return optimal_approximate_shrinkage
        case "optimal_shrinkage": return optimal_shrinkage
        case "bahc": return bahc_filtering
        case _: raise ValueError(f"Invalid covariance cleaner: {cleaner}")
      
#######################
# No Cleaning
#######################

def no_cleaning(
    X: torch.Tensor | numpy.ndarray,
    column_assets: bool =True,
    **kwargs) -> torch.Tensor | numpy.ndarray:

    device = kwargs.get("device", None)
    cov_estimator_params = kwargs["estimator"].get("params", {})
    cov_estimator = get_covariance_estimator(estimator=kwargs["estimator"]["name"])

    X = to_torch_tensor(X, device=device)
    X = X if column_assets else X.T
    cov = cov_estimator(X=X, column_assets=True, device=device, **cov_estimator_params)
    return cov

#######################
# Clipping
####################### 
                
def clipping(
    X: torch.Tensor | numpy.ndarray,
    column_assets: bool = True,
    **kwargs) -> torch.Tensor | numpy.ndarray:
    """
       Reference
       ---------
       "Financial Applications of Random Matrix Theory: a short review",
       J.-P. Bouchaud and M. Potters
       arXiv: 0910.1205 [q-fin.ST]
    """
    cov_estimator_params = kwargs["estimator"].get("params", {})
    cov_estimator = get_covariance_estimator(estimator=kwargs["estimator"]["name"])
    
    device = kwargs.get("device", None)
    
    alpha = kwargs.get("alpha", None)
    assert alpha is None or 0 <= alpha <= 1, f"alpha must be in [0, 1] but got {alpha}"
    
    def marcenko_pastur_lambda_max(n_assets: int, n_observations: int) -> float:
        """
        Reference
        ---------
        "DISTRIBUTION OF EIGENVALUES FOR SOME SETS OF RANDOM MATRICES",
        V. A. Marcenko and L. A. Pastur
        Mathematics of the USSR-Sbornik, Vol. 1 (4), pp 457-483
        """
        assert n_assets > 0
        assert n_observations > 0
        lambda_max = (1 + np.sqrt(n_assets / n_observations))**2
        return lambda_max
    
    X = to_torch_tensor(X, device=device)
    X = X if column_assets else X.T
    n_observations = X.shape[0]
    n_assets = X.shape[1]
    
    # Raw covariance and correlation estimates
    cov  = cov_estimator(X=X, column_assets=True, device=device, **cov_estimator_params)
    corr = corr_estimator(cov=cov, column_assets=True, device=device)
    
    # Correlation matrix eigenvalues and eigenvectors
    eigvals, eigvects = torch.linalg.eigh(corr)

    if alpha is None:
        lambda_max = marcenko_pastur_lambda_max(
            n_assets=n_assets, n_observations=n_observations)
        clipped_eigvals = eigvals.where(eigvals >= lambda_max, torch.nan)
    else:
        clipped_eigvals = to_torch_tensor(
            torch.full(n_assets, torch.nan), device=device)
        threshold = math.ceil(alpha * n_assets)
        if threshold > 0:
            clipped_eigvals[-threshold:] = eigvals[-threshold:]

    gamma = corr.trace() - clipped_eigvals.nansum()
    gamma /= max(torch.isnan(clipped_eigvals).sum().item(), 1)
    
    clipped_eigvals = gamma.where(torch.isnan(clipped_eigvals), clipped_eigvals)
 
    corr_clipped = to_torch_tensor(torch.zeros((n_assets, n_assets)), device=device)
    for eigval, eigvect in zip(clipped_eigvals, eigvects.T):
        eigvect = eigvect.reshape(-1, 1)
        corr_clipped += eigval * eigvect.matmul(eigvect.view(1, -1))
      
    # clean the correlation matrix from numerical errors
    corr_clipped = numerr_clean_corr_mat(corr_clipped)

    # recover the covariance matrix and clean it from numerical errors
    # std = cov.diag().sqrt().clamp(1e-2).reshape(-1, 1)
    stds = cov.diag().sqrt().reshape(-1, 1)
    cov_clipped = corr_clipped * stds.matmul(stds.view(1, -1))
    cov_clipped = numerr_clean_cov_mat(cov_clipped)
    
    return cov_clipped

#######################
# Linear Shrinkage
#######################

def linear_shrinkage(X: numpy.ndarray | torch.Tensor, column_assets: bool =True, **kwargs):
    
    if not isinstance(X, numpy.ndarray):
        X = numpy.array(X)
        
    # compute the linear shrinkage covariance matrix
    X = X if column_assets else X.T
    cov, shrinkage_coeff = sklearn_ledoit_wolf(X=X, assume_centered=False)
    
    # clean the covariance matrix from numerical errors
    cov = to_torch_tensor(cov)
    cov = numerr_clean_cov_mat(cov)
    cov = cov.numpy()
    
    return cov

###############################
# Optimal Approximate Shrinkage
###############################

def optimal_approximate_shrinkage(X: numpy.ndarray | torch.Tensor, column_assets: bool =True, **kwargs):  
    if not isinstance(X, numpy.ndarray):
        X = numpy.array(X)
    
    # compute the OAS-cleaned covariance matrix
    X = X if column_assets else X.T
    cov, shrinkage_coeff = sklearn_oas(X=X, assume_centered=False)
    assert cov.shape == (X.shape[1], X.shape[1]), f"covariance matrix must be of shape ({cov.shape})"
    
    # clean the covariance matrix from numerical errors
    cov = to_torch_tensor(cov)
    cov = numerr_clean_cov_mat(cov)
    cov = cov.numpy()
    return cov

#######################
# Optimal Shrinkage
####################### 

def optimal_shrinkage(X: torch.Tensor | numpy.ndarray, column_assets: bool =True, **kwargs) -> torch.Tensor | numpy.ndarray:
    
    cov_estimator_params = kwargs["estimator"].get("params", {})
    cov_estimator = get_covariance_estimator(estimator=kwargs["estimator"]["name"])
    
    device = kwargs.get("device", None)
    
    method = kwargs.get("method", "kernel")
    assert method in ["kernel", "wishart"], f"method must be one of ['kernel', 'wishart'] but got {method}"
    
    def stieltjes(cov: numpy.ndarray | torch.Tensor, z: Complex | torch.Tensor) -> Complex:
        """
        Reference
        ---------
        "Financial Applications of Random Matrix Theory: a short review",
        J.-P. Bouchaud and M. Potters
        arXiv: 0910.1205 [q-fin.ST]
        """
        if isinstance(z, Complex):
            z = torch.complex(z.real, z.imag)
        else:
            assert isinstance(z, torch.Tensor), f"z must be a complex number but got {type(z)}"
            assert z.shape == (1, 1), f"z must be a scalar but got {z.shape}"
            
        z = to_torch_tensor(z, device=cov.device, dtype=torch.complex32)
        return (z - cov.diag()).mean().item()
    
    def compute_xi(x: float, q: float, cov: numpy.ndarray | torch.Tensor) -> float:
        """
        References
        ----------
        * "Rotational invariant estimator for general noisy matrices",
        J. Bun, R. Allez, J.-P. Bouchaud and M. Potters
        arXiv: 1502.06736 [cond-mat.stat-mech]
        * "Cleaning large Correlation Matrices: tools from Random Matrix Theory",
        J. Bun, J.-P. Bouchaud and M. Potters
        arXiv: 1610.08104 [cond-mat.stat-mech]
        """
        n_assets = cov.shape[0]
    
        z = torch.complex(real=x, imag=-1) / torch.sqrt(n_assets)
        z = to_torch_tensor(z, device=cov.device, dtype=torch.complex32)
        s = stieltjes(cov=cov, z=z)
        
        xi = x / (1 - q + q * z * s).abs()**2
        return xi.item()
    
    def compute_gamma(
        x: float,
        q: float,
        n_assets: int,
        lambda_n_assets: float,
        inverse_wishart: bool =False) -> float:
        """
        Reference
        ---------
        "Cleaning large Correlation Matrices: tools from Random Matrix Theory",
         J. Bun, J.-P. Bouchaud and M. Potters
         arXiv: 1610.08104 [cond-mat.stat-mech]
        """
        z = torch.complex(real=x, imag=-1) / torch.sqrt(n_assets)
        
        lambda_plus = (1 + torch.sqrt(q))**2
        lambda_plus /= (1 - torch.sqrt(q))**2
        lambda_plus *= lambda_n_assets
        
        sigma_2 = lambda_n_assets / (1 - torch.sqrt(q))**2

        # gmp defined below stands for the Stieltjes transform of the
        # rescaled Marcenko-Pastur density, evaluated at z
        gmp = z + sigma_2 * (q - 1) - ((z - lambda_n_assets) * (z - lambda_plus)).sqrt()
        gmp /= 2 * q * sigma_2 * z
        
        gamma = sigma_2 * (1 - q + q * z * gmp).abs()**2
    
        if inverse_wishart:
            kappa = 2 * lambda_n_assets / ((1 - q - lambda_n_assets)**2 - 4 * q * lambda_n_assets)
            alpha_s = 1 / (1 + 2 * q * kappa)
            gamma /= x / (1 + alpha_s * (x - 1))
        else: 
            gamma /= x
    
        return gamma.item()
    
    def direct_kernel_method(
        eigvals: torch.Tensor,
        q: float,
        n_observations: int, n_assets: int) -> torch.Tensor:
        """
       References
       ----------
       * "Eigenvectors of some large sample covariance matrix ensembles",
         O. Ledoit and S. Peche (2011)
       * "Direct Nonlinear Shrinkage Estimation of Large-Dimensional Covariance Matrices (September 2017)", 
         O. Ledoit and M. Wolf https://ssrn.com/abstract=3047302 or http://dx.doi.org/10.2139/ssrn.3047302
        """
        
        def pool_adjancent_violators(y):
            """
            PAV uses the pool adjacent violators method to produce a monotonic smoothing of y.
            Translated from matlab by Sean Collins (2006), and part of the EMAP toolbox.
            """
            device = eigvals.device
            
            y = to_torch_tensor(y, device=device)
            v = y.clone()
        
            n_samples = len(y)
            lvls = torch.arange(n_samples, dtype=torch.int32)
            lvlsets = to_torch_tensor(
                torch.stack([lvls, lvls], dim=1), device=device, dtype=torch.int32)
            
            while True:
                deriv = v.diff()
                if (deriv >= 0).all().item():
                    break

                violator = torch.where(deriv < 0)[0]
                start = lvlsets[violator[0], 0].item()
                last = lvlsets[violator[0] + 1, 1].item()
                v[start:last + 1] = v[start:last + 1].sum().item() / (last - start + 1)
                lvlsets[start:last + 1, 0] = start
                lvlsets[start:last + 1, 1] = last
            
            return v
    
        # compute direct kernel estimator
        lambdas = eigvals[max(0, n_assets - n_observations):]
        h = to_torch_tensor([n_observations**-0.35])
        h_squared = h**2
    
        L = lambdas.expand(n_assets, -1).T
        L_transposed = L.T
        L_transposed_squared = h_squared * (L_transposed ** 2)
        
        zeros = to_torch_tensor(torch.zeros((n_assets, n_assets)))
        tmp = torch.sqrt(torch.max(4 * L_transposed_squared - (L - L_transposed)**2, zeros)) / (2 * torch.pi * L_transposed_squared)
        f_tilde = torch.mean(tmp, dim=0)
        
        tmp = torch.sign(L - L_transposed) * torch.sqrt(torch.max((L - L_transposed)**2 - 4 * L_transposed_squared, zeros)) - L + L_transposed
        tmp /= 2 * torch.pi * L_transposed_squared
        Hf_tilde = torch.mean(tmp, dim=1)
        
        d_tilde = None
        if n_assets <= n_observations:
            tmp = (torch.pi * q * lambdas * f_tilde)**2 + (1 - q - torch.pi * q * lambdas * Hf_tilde)**2
            d_tilde = lambdas / tmp
        else:
            Hf_tilde_0 = (1 - torch.sqrt(1 - 4 * h_squared)) / (2 * torch.pi * h_squared) * torch.mean(1. / lambdas)
            d_tilde_0 = 1 / (torch.pi * (n_assets - n_observations) / n_observations * Hf_tilde_0)
            d_tilde_1 = lambdas / ((torch.pi ** 2) * (lambdas ** 2) * (f_tilde ** 2 + Hf_tilde ** 2))
            d_tilde = torch.cat((d_tilde_0.expand(n_assets - n_observations, -1), d_tilde_1), dim=0)
        
        d_hats = pool_adjancent_violators(d_tilde)
    
        return d_hats
    
    #######################
    # Optimal Shrinkage
        
    X = to_torch_tensor(X, device=device)
    X = X if column_assets else X.T
    n_observations = X.shape[0]
    n_assets = X.shape[1]
    q = n_assets / n_observations
    
    # Raw covariance and correlation estimates
    cov  = cov_estimator(X=X, column_assets=True, device=device, **cov_estimator_params)
    corr = corr_estimator(cov=cov, column_assets=True, device=device)
    
    # Correlation matrix eigenvalues and eigenvectors
    eigvals, eigvects = torch.linalg.eigh(corr)
    
    # The smallest empirical eigenvalue, given that the function used to compute
    # the spectrum of a Hermitian or symmetric matrix - namely torch.linalg.eigh 
    # returns the eigenvalues in ascending order.
    lambda_n_assets = eigvals[0].item()
    
    shrinked_eigvals = None
    if method != 'kernel':
        use_inverse_wishart = (method == 'wishart')
        xis = map(lambda x: compute_xi(x, q, corr), eigvals)
        gammas = map(
            lambda x: compute_gamma(
                x=x, q=q, n_assets=n_assets, lambda_n_assets=lambda_n_assets,
                inverse_wishart=use_inverse_wishart),
            eigvals)
        shrinked_eigvals = map(lambda a, b: a * b if b > 1 else a, xis, gammas)
    else:
        shrinked_eigvals = direct_kernel_method(eigvals, q, n_observations, n_assets)
        
    corr_rie = to_torch_tensor(torch.zeros((n_assets, n_assets), dtype=float), device=device)
    for eigval, eigvect in zip(shrinked_eigvals, eigvects.T):
        eigvect = eigvect.reshape(-1, 1)
        corr_rie += eigval * eigvect.matmul(eigvect.view(1, -1))
    
    # clean the correlation matrix from numerical errors
    corr_rie = numerr_clean_corr_mat(corr_rie)
    # recover the covariance matrix
    stds = cov.diag().sqrt().reshape(-1, 1)
    cov_rie = corr_rie * stds.matmul(stds.T)
    cov_rie = numerr_clean_cov_mat(cov_rie)
    return cov_rie

####################################################
# Bootstrap Average linkage Hierarchical Clustering
####################################################

def bahc_filtering(X: numpy.ndarray | torch.Tensor, column_assets: bool =True, **kwargs):
    order = kwargs.get("order", 1)
    n_bootstrap = kwargs.get("n_bootstrap", 100)
    
    if not isinstance(X, numpy.ndarray):
        X = numpy.array(X)
    
    # compute the bahc-cleaned covariance matrix
    X = X.T if column_assets else X
    cov = bahc.filterCovariance(x=X, K=order, Nboot=n_bootstrap)
    
    assert cov.shape == (X.shape[0], X.shape[0]), f"covariance matrix must be of shape ({cov.shape})"
    
    # clean the covariance matrix from numerical errors
    cov = to_torch_tensor(cov)
    cov = numerr_clean_cov_mat(cov)
    cov = cov.numpy()
    return cov
