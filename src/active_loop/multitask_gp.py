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
                 lengthscale: float = None,
                 outputscale: float = None,
                 noise: float = None,
                 lengthscale_prior_mean: float = None,
                 lengthscale_prior_std: float = None,
                 outputscale_prior_mean: float = None,
                 outputscale_prior_std: float = None,
                 noise_prior_mean: float = None,
                 noise_prior_std: float = None,
                 noise_bounds: tuple[float, float] = None,
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
        
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.noise = noise
        
        # Store prior parameters
        self.lengthscale_prior_mean = lengthscale_prior_mean
        self.lengthscale_prior_std = lengthscale_prior_std
        self.outputscale_prior_mean = outputscale_prior_mean
        self.outputscale_prior_std = outputscale_prior_std
        self.noise_prior_mean = noise_prior_mean
        self.noise_prior_std = noise_prior_std
        self.noise_bounds = noise_bounds

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
            
            if self.lengthscale_prior_mean is not None and self.lengthscale_prior_std is not None:
                # Using LogNormal prior for lengthscale (must be positive)
                lengthscale_prior = LogNormalPrior(
                    loc=torch.tensor(np.log(self.lengthscale_prior_mean)),
                    scale=torch.tensor(self.lengthscale_prior_std)
                )
                kernel.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, "lengthscale"
                )
                
            if self.outputscale_prior_mean is not None and self.outputscale_prior_std is not None:
                # Using LogNormal prior for outputscale (must be positive)
                outputscale_prior = LogNormalPrior(
                    loc=torch.tensor(np.log(self.outputscale_prior_mean)),
                    scale=torch.tensor(self.outputscale_prior_std)
                )
                kernel.register_prior(
                    "outputscale_prior", outputscale_prior, "outputscale"
                )
            
            # Set up noise constraints if provided
            noise_constraint = None
            if self.noise_bounds is not None:
                from gpytorch.constraints import Interval
                noise_constraint = Interval(
                    lower_bound=torch.tensor(self.noise_bounds[0]),
                    upper_bound=torch.tensor(self.noise_bounds[1])
                )
            
            # Create the likelihood with appropriate constraints
            likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
            
            # Initialize with provided noise value or use median of train_std
            if self.noise is not None:
                likelihood.noise = torch.tensor(self.noise)
            else:
                init_noise = std_dim.pow(2).median().item()
                if self.noise_bounds is not None:
                    init_noise = np.clip(init_noise, self.noise_bounds[0] + 1e-6, 
                                         self.noise_bounds[1] - 1e-6)
                try:
                    likelihood.noise = torch.tensor(init_noise)
                except RuntimeError as e:
                    print(f"Error setting noise: {e}")
                    likelihood.noise = torch.tensor((self.noise_bounds[0] + self.noise_bounds[1]) / 2)
            
            # Add noise prior if specified
            if self.noise_prior_mean is not None and self.noise_prior_std is not None:
                noise_prior = LogNormalPrior(
                    loc=torch.tensor(np.log(self.noise_prior_mean)),
                    scale=torch.tensor(self.noise_prior_std)
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
                    kernel.base_kernel.raw_lengthscale.data.fill_(self.lengthscale)
                if self.outputscale is not None:
                    kernel.raw_outputscale.data.fill_(self.outputscale)
            
            # Add the fitted GP to our list
            self.gp_models.append(gp)

    def acq(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the acquisition function for a given set of points.
        For multitask GPs, we sum the acquisition values across all dimensions.
        
        Args:
            x: Points to evaluate
            
        Returns:
            Acquisition function values
        """
        if not self.gp_models:
            raise ValueError("GP models are not fitted yet.")
        
        # Compute acquisition function for each dimension and sum
        acq_values = torch.zeros(x.shape[0], device=x.device)
        
        for gp in self.gp_models:
            acq = UncertaintyAcquisition(gp)
            acq_values += acq(x)
        
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
        
        # Define a combined acquisition function that sums over all dimensions
        acq_func = MultitaskUncertaintyAcquisition(self.gp_models)
        
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
    Combined acquisition function for multitask GPs that sums uncertainties across all tasks.
    """
    def __init__(self, models):
        # Use the first model for inheritance requirements
        super().__init__(model=models[0])
        self.models = models
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Sum uncertainties from all models
        total_uncertainty = torch.zeros(X.shape[0], device=X.device)
        
        for model in self.models:
            posterior = model.posterior(X)
            uncertainty = posterior.variance.squeeze((-1, -2))
            total_uncertainty += uncertainty
        
        return total_uncertainty


if __name__ == "__main__":
    # Example usage
    active_measurement = MultitaskActiveMeasurement(
        initial_points=[-24, 0, 24],
        param_indices=(0, 1),  # Track first two parameters
        bounds=[-24, 24],
        fit_kernel=True,
        max_num_points=30,
        lengthscale_prior_mean=1.0,
        lengthscale_prior_std=0.3,
        outputscale_prior_mean=1.0,
        outputscale_prior_std=0.3,
        noise_prior_mean=0.01,
        noise_prior_std=0.5,
        noise_bounds=(1e-6, 0.1),
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
