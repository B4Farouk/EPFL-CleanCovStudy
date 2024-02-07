import torch
import numpy

from typing import Optional

from utils import to_torch_tensor

from covariance.numerr_cleaning import numerr_clean_corr_mat, numerr_clean_cov_mat

def get_covariance_estimator(estimator: str) -> callable:
    match estimator:
        case "natural": return cov_estimator
        case "ewm1": return ewm1_cov_estimator
        case _ : raise ValueError(f"Invalid covariance estimator: {estimator}")
        
def cov_estimator_is_regressive(estimator: str) -> bool:
    match estimator:
        case "natural": return False
        case "ewm1": return True
        case _ : raise ValueError(f"Invalid covariance estimator: {estimator}")

def cov_estimator(
    X: numpy.ndarray | torch.Tensor,
    column_assets: bool =True,
    device: str =None,
    **kwargs) -> numpy.ndarray | torch.Tensor:
    """Unbiased covariance estimator
    """
    assert X.dim() == 2, "X must be a 2D array"
    
    X = to_torch_tensor(X, device=device)
    X = X if column_assets else X.T
    
    n_observations = X.shape[0]
    assert n_observations > 1, "X must have at least 2 observations"
    
    # Compute covariance matrix
    X_centered = X - X.mean(dim=0)
    cov = (X_centered.T).matmul(X_centered)    
    cov /= (n_observations - 1)
    
    cov = numerr_clean_cov_mat(cov)
    return cov


def corr_estimator(
    X: Optional[numpy.ndarray | torch.Tensor] =None,
    cov: Optional[numpy.ndarray | torch.Tensor] =None,
    column_assets: bool =True,
    device: str =None) -> numpy.ndarray | torch.Tensor:
    """Unbiased correlation estimator
    """
    assert X is None or cov is None, "Only one of X or cov can be specified"
    
    if cov is None:
        cov = cov_estimator(X, column_assets=column_assets, device=device)
    else:
        cov = to_torch_tensor(cov, device=device)
    
    # Compute correlation matrix
    stds = cov.diag().sqrt().reshape(-1, 1)
    assert not (stds == 0.0).any().item(), "At least one asset has null standard deviation. Correlation matrix is undefined."
    corr = cov / stds.matmul(stds.view(1, -1))
    
    corr = numerr_clean_corr_mat(corr)
    return corr


def ewm1_cov_estimator(X: numpy.ndarray | torch.Tensor,
                       column_assets: bool =True,
                       device: str =None,
                       **kwargs) -> numpy.ndarray | torch.Tensor:
    """Exponential weighted moving average covariance estimator
    """
    cov_prev = kwargs["cov_prev"]
    delta_t = kwargs["delta_t"]
    decay_params = kwargs.get("decay_params", {"mode": "exponential", "decay": 0.01})
    target_assets_mask = kwargs.get("target_assets_mask", None)
    
    # Check inputs: data
    assert X.shape[0] > 1, "X must have at least 2 observations"
    assert X.dim() == 2, "X must be a 2D array"
    # Check inputs: previous covariance matrix
    assert (cov_prev is None) or (cov_prev.dim() == 2 and cov_prev.shape[0] == cov_prev.shape[1]), "cov_prev must be a 2D array"
    # Check inputs: parameters
    mode = decay_params["mode"] 
    assert mode in ["exponential", "linear"], f"Invalid EWM-1 covariance estimation mode {mode}"
    assert delta_t > 0, "delta_t must be positive"
    decay = decay_params.get("decay", None)
    halflife = decay_params.get("halflife", None)
    assert not(halflife is not None and decay is not None), "Only one of halflife or decay can be specified"
    assert (halflife is not None and halflife > 0) or (decay is not None and decay > 0)
    
    # Decay:
    # Decay is zero if no previous covariance matrix is provided
    decay_factor = 0.0
    if cov_prev is not None:
        match mode:
            case "exponential":
                halflife = halflife if halflife is not None else numpy.log(2) / decay
                decay_factor = 2 ** (- delta_t / halflife)
            case "linear":
                halflife = halflife if halflife is not None else 1 / (2 * decay)
                decay_factor = max(1 - delta_t / halflife, 0.0)
            case _: raise ValueError(f"Invalid mode: {mode}")
            
    # Covariance matrix
    cov = cov_estimator(X, column_assets=column_assets, device=device)
    # Get or initialize the previous covariance matrix
    if cov_prev is None:
        n_all_assets = (
            len(target_assets_mask)
            if target_assets_mask is not None
            else X.shape[1]
        )
        cov_prev = to_torch_tensor(torch.zeros_like(torch.zeros(n_all_assets, n_all_assets)), device=device)
    else:
        cov_prev = to_torch_tensor(cov_prev, device=device)
          
    # We use the target assets mask to perform targetted updates on specific assets.
    # In particular, if two out of four assets are targetted, only the covariances
    # of these assets will be updated.
    # This allows us to keep a fixed size covariance matrix (cov_global) for future
    # strategy steps in which a different number of assets may be considered.
    to_be_updated = None
    if target_assets_mask is not None:
        # Case not all assets are present
        target_assets_mask_t = to_torch_tensor(torch.BoolTensor(target_assets_mask).view((-1, 1)), device=device)
        to_be_updated = (target_assets_mask_t).matmul(target_assets_mask_t.view((1, -1)))
        to_be_updated_rows, to_be_updated_cols = torch.nonzero(to_be_updated, as_tuple=True)
        print(torch.sum(to_be_updated).item(), sum(target_assets_mask))
        assert torch.sum(to_be_updated).item() == sum(target_assets_mask), "The update mask is inconsistent with the number of changed assets."

    # Update step
    if to_be_updated is not None:
        cov_prev[to_be_updated_rows, to_be_updated_cols] = decay_factor * cov_prev[to_be_updated_rows, to_be_updated_cols] + (1.0 - decay_factor) * cov
        cov_prev = numerr_clean_cov_mat(cov_prev)
    else:
        cov_prev = decay_factor * cov_prev + (1.0 - decay_factor) * cov
        cov_prev = numerr_clean_cov_mat(cov_prev)
        
    return cov, cov_prev
