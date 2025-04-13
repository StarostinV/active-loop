from typing import Optional, Tuple, List, Union

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import LogNormalPrior

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from active_loop.active_measurement import ActiveMeasurement


class MultitaskActiveMeasurement(ActiveMeasurement):
    def __init__(self, 
                 initial_points: list[float],
                 param_indices: Tuple[int, ...],
                 gpkernel: Kernel = None,
                 bounds: tuple[float, float] = (0, 1),
                 fit_kernel: bool = True,
                 max_num_points: int = 30,
                 acquisition_weights: Union[List[float], None] = None,
                 lengthscale: Union[float, List[float]] = None,
                 outputscale: Union[float, List[float]] = None,
                 noise: Union[float, List[float]] = None,
                 lengthscale_prior_mean: Union[float, List[float]] = None,
                 lengthscale_prior_std: Union[float, List[float]] = None,
                 outputscale_prior_mean: Union[float, List[float]] = None,
                 outputscale_prior_std: Union[float, List[float]] = None,
                 noise_prior_mean: Union[float, List[float]] = None,
                 noise_prior_std: Union[float, List[float]] = None,
                 noise_bounds: Union[tuple[float, float], List[tuple[float, float]]] = None,
                 ):
        # Determine ndim from param_indices
        ndim = len(param_indices)
        
        super().__init__(
            initial_points=initial_points,
            bounds=bounds,
            ndim=ndim,
            max_num_points=max_num_points,
        )
        
        self.param_indices = param_indices
        self.fit_kernel = fit_kernel
        
        # Set acquisition weights for combining uncertainties across dimensions
        if acquisition_weights is None:
            # Default to equal weights
            self.acquisition_weights = torch.ones(ndim)
        else:
            if len(acquisition_weights) != ndim:
                raise ValueError(f"Length of acquisition_weights ({len(acquisition_weights)}) must match ndim ({ndim})")
            self.acquisition_weights = torch.tensor(acquisition_weights)
            # Normalize weights to sum to ndim (to maintain same scale as unweighted case)
            self.acquisition_weights = self.acquisition_weights * (ndim / self.acquisition_weights.sum())
        
        # Initialize a separate GP model for each output dimension
        self.gp_models = []
        
        # Create a separate kernel for each output dimension if not provided
        if gpkernel is None:
            self.gpkernels = [ScaleKernel(RBFKernel()) for _ in range(ndim)]
        elif isinstance(gpkernel, list):
            self.gpkernels = gpkernel
        else:
            # If a single kernel is provided, use it for all dimensions
            self.gpkernels = [gpkernel] * ndim
        
        # Store parameters, ensuring they are proper lists for each dimension
        self.lengthscale = self._ensure_list_param(lengthscale, ndim)
        self.outputscale = self._ensure_list_param(outputscale, ndim)
        self.noise = self._ensure_list_param(noise, ndim)
        
        # Store prior parameters
        self.lengthscale_prior_mean = self._ensure_list_param(lengthscale_prior_mean, ndim)
        self.lengthscale_prior_std = self._ensure_list_param(lengthscale_prior_std, ndim)
        self.outputscale_prior_mean = self._ensure_list_param(outputscale_prior_mean, ndim)
        self.outputscale_prior_std = self._ensure_list_param(outputscale_prior_std, ndim)
        self.noise_prior_mean = self._ensure_list_param(noise_prior_mean, ndim)
        self.noise_prior_std = self._ensure_list_param(noise_prior_std, ndim)
        
        # Handle noise bounds
        if noise_bounds is None:
            self.noise_bounds = None
        elif isinstance(noise_bounds[0], (list, tuple)) and len(noise_bounds) == ndim:
            # Already a list of bounds
            self.noise_bounds = noise_bounds
        else:
            # Single bound for all dimensions
            self.noise_bounds = [noise_bounds] * ndim

    def _ensure_list_param(self, param, ndim):
        """
        Ensure parameter is a list with length matching the number of dimensions.
        
        Args:
            param: Parameter (can be None, single value, or list)
            ndim: Number of dimensions
            
        Returns:
            List of parameter values, or None if param is None
        """
        if param is None:
            return None
        elif isinstance(param, (list, tuple)):
            if len(param) != ndim:
                raise ValueError(f"Parameter list length ({len(param)}) does not match number of dimensions ({ndim})")
            return list(param)
        else:
            # Single value for all dimensions
            return [param] * ndim

    def preprocess_input(self, x, y, y_std):
        """
        Preprocess inputs for fitting GP models.
        
        Args:
            x: Input features (1D)
            y: Target values (multi-dimensional)
            y_std: Standard deviations of target values (multi-dimensional)
        
        Returns:
            Processed x, y, y_std
        """
        # Ensure consistent dtype (double/float64)
        x = torch.atleast_1d(x.flatten()).to(dtype=torch.double)[:, None]  # Make x 2D
        
        # Handle y dimensions based on param_indices
        if len(y.shape) == 1:
            # If y is 1D, assume it has all dimensions stacked
            y_processed = []
            y_std_processed = []
            
            for idx in self.param_indices:
                y_processed.append(torch.atleast_1d(y[idx]).to(dtype=torch.double).unsqueeze(-1))
                y_std_processed.append(torch.atleast_1d(y_std[idx]).to(dtype=torch.double).unsqueeze(-1))
                
            return x, y_processed, y_std_processed
        else:
            # If y is already multi-dimensional, select the relevant indices
            y_processed = []
            y_std_processed = []
            
            for idx in self.param_indices:
                y_processed.append(torch.atleast_1d(y[:, idx]).to(dtype=torch.double).unsqueeze(-1))
                y_std_processed.append(torch.atleast_1d(y_std[:, idx]).to(dtype=torch.double).unsqueeze(-1))
                
            return x, y_processed, y_std_processed
    
    def add_points(self, x: torch.Tensor, y: torch.Tensor, y_std: torch.Tensor, update_gp: bool = True):
        """
        Add new data points to the training set and optionally update the GP models.
        
        Args:
            x: Input features
            y: Target values
            y_std: Standard deviations
            update_gp: Whether to update the GP models after adding points
        """
        x, y_processed, y_std_processed = self.preprocess_input(x, y, y_std)
        
        # Concatenate with existing data
        self.train_x = torch.cat([self.train_x, x])
        
        # For multidimensional data, we need to handle concatenation differently
        if self.train_y.shape[0] == 0:
            # First data point being added
            stacked_y = torch.cat([y_dim for y_dim in y_processed], dim=1)
            stacked_y_std = torch.cat([y_std_dim for y_std_dim in y_std_processed], dim=1)
            self.train_y = stacked_y
            self.train_y_std = stacked_y_std
        else:
            # Subsequent data points
            stacked_y = torch.cat([y_dim for y_dim in y_processed], dim=1)
            stacked_y_std = torch.cat([y_std_dim for y_std_dim in y_std_processed], dim=1)
            self.train_y = torch.cat([self.train_y, stacked_y])
            self.train_y_std = torch.cat([self.train_y_std, stacked_y_std])
        
        if update_gp:
            self.update_gp(self.train_x, self.train_y, self.train_y_std)
    
    def update_gp(self, train_x: torch.Tensor, train_y: torch.Tensor, train_std: torch.Tensor):
        """
        Update all GP models with the current training data.
        
        Args:
            train_x: Training input features
            train_y: Training target values
            train_std: Training standard deviations
        """
        # Clear existing models
        self.gp_models = []
        
        # For each output dimension, create and fit a separate GP model
        for dim in range(self.ndim):
            # Get the training data for this dimension
            y_dim = train_y[:, dim:dim+1]
            std_dim = train_std[:, dim:dim+1]
            
            # Configure the kernel with priors if specified
            kernel = self.gpkernels[dim]
            
            # Apply dimension-specific lengthscale prior if available
            if self.lengthscale_prior_mean is not None and self.lengthscale_prior_std is not None:
                # Get dimension-specific values
                ls_mean = self.lengthscale_prior_mean[dim]
                ls_std = self.lengthscale_prior_std[dim]
                
                # Using LogNormal prior for lengthscale (must be positive)
                lengthscale_prior = LogNormalPrior(
                    loc=torch.tensor(np.log(ls_mean)),
                    scale=torch.tensor(ls_std)
                )
                kernel.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, "lengthscale"
                )
                
            # Apply dimension-specific outputscale prior if available
            if self.outputscale_prior_mean is not None and self.outputscale_prior_std is not None:
                # Get dimension-specific values
                os_mean = self.outputscale_prior_mean[dim]
                os_std = self.outputscale_prior_std[dim]
                
                # Using LogNormal prior for outputscale (must be positive)
                outputscale_prior = LogNormalPrior(
                    loc=torch.tensor(np.log(os_mean)),
                    scale=torch.tensor(os_std)
                )
                kernel.register_prior(
                    "outputscale_prior", outputscale_prior, "outputscale"
                )
            
            # Set up noise constraints if provided
            noise_constraint = None
            if self.noise_bounds is not None:
                from gpytorch.constraints import Interval
                # Get dimension-specific noise bounds
                noise_bounds = self.noise_bounds[dim]
                noise_constraint = Interval(
                    lower_bound=torch.tensor(noise_bounds[0]),
                    upper_bound=torch.tensor(noise_bounds[1])
                )
            
            # Create the likelihood with appropriate constraints
            likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
            
            # Initialize with provided noise value or use median of train_std
            if self.noise is not None:
                # Use dimension-specific noise if available
                likelihood.noise = torch.tensor(self.noise[dim])
            else:
                init_noise = std_dim.pow(2).median().item()
                if self.noise_bounds is not None:
                    noise_bounds = self.noise_bounds[dim]
                    init_noise = np.clip(init_noise, noise_bounds[0] + 1e-6, 
                                         noise_bounds[1] - 1e-6)
                try:
                    likelihood.noise = torch.tensor(init_noise)
                except RuntimeError as e:
                    print(f"Error setting noise for dim {dim}: {e}")
                    noise_bounds = self.noise_bounds[dim]
                    likelihood.noise = torch.tensor((noise_bounds[0] + noise_bounds[1]) / 2)
            
            # Add dimension-specific noise prior if specified
            if self.noise_prior_mean is not None and self.noise_prior_std is not None:
                # Get dimension-specific values
                noise_mean = self.noise_prior_mean[dim]
                noise_std = self.noise_prior_std[dim]
                
                noise_prior = LogNormalPrior(
                    loc=torch.tensor(np.log(noise_mean)),
                    scale=torch.tensor(noise_std)
                )
                likelihood.register_prior("noise_prior", noise_prior, "noise")
            
            # Create GP model
            gp = SingleTaskGP(
                train_X=train_x,
                train_Y=y_dim,
                covar_module=kernel,
                likelihood=likelihood
            )
            
            # Fit the GP if requested
            if self.fit_kernel:
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
            else:
                if self.lengthscale is not None:
                    # Use dimension-specific lengthscale if available
                    kernel.base_kernel.raw_lengthscale.data.fill_(self.lengthscale[dim])
                if self.outputscale is not None:
                    # Use dimension-specific outputscale if available
                    kernel.raw_outputscale.data.fill_(self.outputscale[dim])
            
            # Add the fitted GP to our list
            self.gp_models.append(gp)

    def acq(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the acquisition function for a given set of points.
        For multitask GPs, we compute a weighted sum of acquisition values across dimensions.
        
        Args:
            x: Points to evaluate
            
        Returns:
            Acquisition function values
        """
        if not self.gp_models:
            raise ValueError("GP models are not fitted yet.")
        
        # Compute weighted acquisition function for each dimension
        acq_values = torch.zeros(x.shape[0], device=x.device)
        
        for i, gp in enumerate(self.gp_models):
            acq = UncertaintyAcquisition(gp)
            # Apply weight for this dimension
            acq_values += self.acquisition_weights[i] * acq(x)
        
        return acq_values

    def optimize_acq(self, q: int = 1, num_restarts: int = 20, raw_samples: int = 100) -> torch.Tensor:
        """
        Optimize the acquisition function to find the next best point(s) to sample.
        
        Args:
            q: Number of points to select
            num_restarts: Number of optimization restarts
            raw_samples: Number of initial samples
            
        Returns:
            Selected points
        """
        if not self.gp_models:
            raise ValueError("GP models are not fitted yet.")
        
        # Define a combined acquisition function that applies weights
        acq_func = MultitaskUncertaintyAcquisition(self.gp_models, self.acquisition_weights)
        
        return optimize_acqf(
            acq_function=acq_func, 
            bounds=self.bounds, 
            q=q, 
            num_restarts=num_restarts, 
            raw_samples=raw_samples
        )
    
    def find_candidate(self, q: int = 1, num_restarts: int = 20, raw_samples: int = 100) -> torch.Tensor:
        """
        Find the next best candidate point(s) to sample.
        
        Args:
            q: Number of points to select
            num_restarts: Number of optimization restarts
            raw_samples: Number of initial samples
            
        Returns:
            Selected point(s)
        """
        return self.optimize_acq(q=q, num_restarts=num_restarts, raw_samples=raw_samples)[0]

    def plot_gp(self, test_x: Optional[torch.Tensor] = None):
        """
        Plot all GP models in a stacked layout.
        
        Args:
            test_x: Optional test points to use for plotting
        """
        if not self.gp_models:
            raise ValueError("GP models are not fitted yet.")
        
        if test_x is None:
            test_x = torch.linspace(self.x_min, self.x_max, 100).unsqueeze(-1)
        
        # Create a figure with subplots, one for each dimension
        fig, axes = plt.subplots(nrows=self.ndim, figsize=(10, 4 * self.ndim))
        if self.ndim == 1:
            axes = [axes]
        
        # Plot each GP model
        for dim, (gp, ax) in enumerate(zip(self.gp_models, axes)):
            with torch.no_grad():
                posterior = gp.posterior(test_x)
                mean = posterior.mean
                lower, upper = posterior.mvn.confidence_region()
                
                ax.plot(test_x.numpy(), mean.numpy(), 'b-', label='GP mean')
                ax.fill_between(test_x.squeeze().numpy(), 
                               lower.numpy(), upper.numpy(), 
                               alpha=0.2, color='b', label='GP uncertainty')
                
                # Plot training data for this dimension
                ax.scatter(self.train_x.numpy(), self.train_y[:, dim].numpy(), 
                          c='r', marker='x', label='Measurements')
                
                ax.set_title(f'Dimension {dim} (Index: {self.param_indices[dim]})')
                ax.legend()
        
        plt.tight_layout()
        return fig


class UncertaintyAcquisition(AnalyticAcquisitionFunction):
    """
    Acquisition function based on prediction uncertainty.
    """
    def __init__(self, model):
        super().__init__(model=model)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X)
        res = posterior.variance.squeeze((-1, -2))
        return res


class MultitaskUncertaintyAcquisition(AnalyticAcquisitionFunction):
    """
    Combined acquisition function for multitask GPs that computes a weighted sum
    of uncertainties across all tasks.
    """
    def __init__(self, models, weights=None):
        # Use the first model for inheritance requirements
        super().__init__(model=models[0])
        self.models = models
        
        # Set weights for combining uncertainties
        if weights is None:
            # Equal weights if none provided
            self.weights = torch.ones(len(models))
        else:
            self.weights = weights
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Sum weighted uncertainties from all models
        total_uncertainty = torch.zeros(X.shape[0], device=X.device)
        
        for i, model in enumerate(self.models):
            posterior = model.posterior(X)
            uncertainty = posterior.variance.squeeze((-1, -2))
            total_uncertainty += self.weights[i] * uncertainty
        
        return total_uncertainty


if __name__ == "__main__":
    # Example usage with dimension-specific priors and acquisition weights
    active_measurement = MultitaskActiveMeasurement(
        initial_points=[-24, 0, 24],
        param_indices=(0, 1),  # Track first two parameters
        bounds=[-24, 24],
        fit_kernel=True,
        max_num_points=30,
        # Weigh uncertainty of first dimension 2x more than second dimension
        acquisition_weights=[2.0, 1.0],  
        # Different priors for each dimension
        lengthscale_prior_mean=[1.0, 0.5],  # First dimension has longer lengthscale prior
        lengthscale_prior_std=[0.3, 0.2],
        outputscale_prior_mean=[1.0, 2.0],  # Second dimension has higher variance prior
        outputscale_prior_std=[0.3, 0.4],
        noise_prior_mean=[0.01, 0.02],      # Different noise levels per dimension
        noise_prior_std=[0.5, 0.5],
        noise_bounds=[(1e-6, 0.1), (1e-6, 0.2)],  # Different noise bounds per dimension
    )
    
    # Add initial data points - ensure all tensors are double type
    active_measurement.add_points(
        torch.tensor([-24], dtype=torch.double), 
        torch.tensor([[0.5, 0.2]], dtype=torch.double),  # Two-dimensional output 
        torch.tensor([[0.01, 0.01]], dtype=torch.double), 
        update_gp=False
    )
    
    active_measurement.add_points(
        torch.tensor([0], dtype=torch.double), 
        torch.tensor([[0.0, 0.3]], dtype=torch.double), 
        torch.tensor([[0.01, 0.01]], dtype=torch.double), 
        update_gp=True
    )
    
    active_measurement.add_points(
        torch.tensor([24], dtype=torch.double), 
        torch.tensor([[-0.3, -0.1]], dtype=torch.double), 
        torch.tensor([[0.01, 0.01]], dtype=torch.double), 
        update_gp=True
    )
    
    # Find next candidate and plot
    candidate = active_measurement.find_candidate()
    print("Next candidate:", candidate)
    
    active_measurement.plot_gp()
    plt.savefig("multitask_gp.png")
