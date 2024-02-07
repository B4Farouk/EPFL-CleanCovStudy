import torch
import numpy

def symmetrize(mat: numpy.ndarray | torch.Tensor) -> numpy.ndarray | torch.Tensor:
    mat = (mat + mat.T.conj()) / 2.0
    return mat

def numerr_clean_cov_mat(cov: numpy.ndarray | torch.Tensor) -> numpy.ndarray | torch.Tensor:
    cov = symmetrize(cov)
    assert torch.allclose(cov, cov.T.conj()), "Covariance matrix must be symmetric"
    return cov

def numerr_clean_corr_mat(corr: torch.Tensor) -> numpy.ndarray | torch.Tensor:
    corr = corr.clamp(-1.0, 1.0)
    corr = symmetrize(corr)
    
    sqrt_corr_diag = corr.diag().abs().sqrt().reshape(-1, 1)
    assert not sqrt_corr_diag.isnan().any().item(), "Correlation matrix diagonal must not contain NaNs"
    
    corr /= sqrt_corr_diag.matmul(sqrt_corr_diag.T)
    corr = corr.clamp(-1.0, 1.0)
    corr = symmetrize(corr)
    
    assert torch.allclose(corr, corr.T.conj()), "Correlation matrix must be symmetric"
    return corr