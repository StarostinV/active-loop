from typing import Optional

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel, ProductKernel
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.priors import NormalPrior, LogNormalPrior

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import AnalyticAcquisitionFunction, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from active_loop.active_measurement import ActiveMeasurement


class Spatial2DimActiveMeasurement(ActiveMeasurement):
    def __init__(self, 
                 initial_points: list[list[float]],
                 gpkernel: Optional[Kernel] = None,
                 bounds: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 param_idx: int = 0,
                 fit_kernel: bool = True,
                 fit_likelihood: bool = False,
                 max_num_points: int = 30,
                 lengthscale: Optional[list[float]] = None,
                 outputscale: Optional[float] = None,
                 noise: Optional[float] = None,
                 lengthscale_prior_mean: Optional[list[float]] = None,
                 lengthscale_prior_std: Optional[list[float]] = None,
                 outputscale_prior_mean: Optional[float] = None,
                 outputscale_prior_std: Optional[float] = None,
                 noise_prior_mean: Optional[float] = None,
                 noise_prior_std: Optional[float] = None,
                 noise_bounds: Optional[tuple[float, float]] = None,
                 axis_names: tuple[str, str] = ("x", "om"),
                 ):
        """
        Initialize SpatialDimActiveMeasurement with 2D spatial inputs.
        
        Args:
            initial_points: List of initial 2D points [[x1, y1], [x2, y2], ...]
            gpkernel: Optional custom kernel
            bounds: Tuple of bounds for each dimension ((x_min, x_max), (y_min, y_max))
            param_idx: Index of the parameter to optimize
            fit_kernel: Whether to fit kernel hyperparameters
            fit_likelihood: Whether to fit noise parameter instead of using fixed noise
            max_num_points: Maximum number of points to collect
            lengthscale: Optional list of initial lengthscales for each dimension [ls_x, ls_y]
            outputscale: Optional initial outputscale
            noise: Optional initial noise value when fit_likelihood=True
            lengthscale_prior_mean: Optional list of prior means for lengthscales [mean_x, mean_y]
            lengthscale_prior_std: Optional list of prior stds for lengthscales [std_x, std_y]
            outputscale_prior_mean: Optional prior mean for outputscale
            outputscale_prior_std: Optional prior std for outputscale
            noise_prior_mean: Optional prior mean for noise (in log space)
            noise_prior_std: Optional prior std for noise (in log space)
            noise_bounds: Optional tuple of (min, max) bounds for noise
            axis_names: Optional tuple of (x_name, y_name) axis names. 
                Default is ("x", "om").
        """
        # Convert bounds to flatten for the base class
        
        super().__init__(
            initial_points=initial_points,
            bounds=bounds,
            ndim=1,
            xdim=2,
            max_num_points=max_num_points,
        )
        
        # Create default kernel if none provided
        if gpkernel is None:
            # Create a product kernel with two RBF kernels, one for each dimension
            # This allows for different lengthscales per dimension
            dim1_kernel = ScaleKernel(RBFKernel(ard_num_dims=2))
            gpkernel = dim1_kernel
            
        self.gpkernel = gpkernel
        self.gp = None
        self.param_idx = param_idx
        self.fit_kernel = fit_kernel
        self.fit_likelihood = fit_likelihood

        # Store kernel hyperparameters
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
        self.axis_names = axis_names

    def preprocess_input(self, x, y, y_std):
        """Preprocess inputs for the GP"""
        # Ensure x is a 2D tensor [batch_size, 2]
        if x.dim() == 1 and x.size(0) == 2:
            # Single point as [x, y]
            x = x.unsqueeze(0)
        elif x.dim() == 2 and x.size(1) != 2:
            raise ValueError(f"Input x must have shape [batch_size, 2], got {x.shape}")
            
        # Process y and y_std
        if y.dim() > 1:
            y = y.flatten()[self.param_idx].unsqueeze(0)
        else:
            y = y[self.param_idx].unsqueeze(0)
            
        if y_std.dim() > 1:
            y_std = y_std.flatten()[self.param_idx].unsqueeze(0)
        else:
            y_std = y_std[self.param_idx].unsqueeze(0)
            
        return x, y.unsqueeze(-1), y_std.unsqueeze(-1)
    
    def add_points(self, x: torch.Tensor, y: torch.Tensor, y_std: torch.Tensor, update_gp: bool = True):
        """Add new observation points and update the GP model"""
        x, y, y_std = self.preprocess_input(x, y, y_std)
        self.train_x = torch.cat([self.train_x, x])
        self.train_y = torch.cat([self.train_y, y])
        self.train_y_std = torch.cat([self.train_y_std, y_std])
        
        if update_gp:
            self.update_gp(self.train_x, self.train_y, self.train_y_std)
    
    def update_gp(self,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            train_std: torch.Tensor,
            ):
        """
        Fit the GP model to the training data.
        
        Args:
            train_x: Training input points (2D spatial points)
            train_y: Training target values
            train_std: Point-dependent noise levels
        """
        # Configure kernel with priors if specified
        if self.lengthscale_prior_mean is not None and self.lengthscale_prior_std is not None:
            # Using LogNormal prior for lengthscale (must be positive)
            # For ARD kernel, we need to set priors for each dimension
            if isinstance(self.lengthscale_prior_mean, list) and isinstance(self.lengthscale_prior_std, list):
                # Create priors for each dimension
                for i, (mean, std) in enumerate(zip(self.lengthscale_prior_mean, self.lengthscale_prior_std)):
                    lengthscale_prior = LogNormalPrior(
                        loc=torch.tensor(np.log(mean)),
                        scale=torch.tensor(std)
                    )
                    # If using ARD kernel, register prior for each lengthscale dimension
                    if hasattr(self.gpkernel.base_kernel, 'raw_lengthscale') and self.gpkernel.base_kernel.raw_lengthscale.size(-1) > 1:
                        name = f"lengthscale_prior_{i}"
                        self.gpkernel.base_kernel.register_prior(
                            name, lengthscale_prior, lambda module, i=i: module.lengthscale[..., i]
                        )
            else:
                # If not provided as list, use the same prior for all dimensions
                lengthscale_prior = LogNormalPrior(
                    loc=torch.tensor(np.log(self.lengthscale_prior_mean)),
                    scale=torch.tensor(self.lengthscale_prior_std)
                )
                self.gpkernel.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, "lengthscale"
                )
            
        if self.outputscale_prior_mean is not None and self.outputscale_prior_std is not None:
            # Using LogNormal prior for outputscale (must be positive)
            outputscale_prior = LogNormalPrior(
                loc=torch.tensor(np.log(self.outputscale_prior_mean)),
                scale=torch.tensor(self.outputscale_prior_std)
            )
            self.gpkernel.register_prior(
                "outputscale_prior", outputscale_prior, "outputscale"
            )
        
        # Choose between fixed noise or fitted likelihood
        if self.fit_likelihood:
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
                init_noise = train_std.pow(2).median().item()
                likelihood.noise = torch.tensor(init_noise)
            
            # Add noise prior if specified
            if self.noise_prior_mean is not None and self.noise_prior_std is not None:
                noise_prior = LogNormalPrior(
                    loc=torch.tensor(np.log(self.noise_prior_mean)),
                    scale=torch.tensor(self.noise_prior_std)
                )
                likelihood.register_prior("noise_prior", noise_prior, "noise")
            
            self.gp = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                covar_module=self.gpkernel,
                likelihood=likelihood
            )
        else:
            # Use traditional fixed noise approach
            self.gp = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                covar_module=self.gpkernel,
                likelihood=FixedNoiseGaussianLikelihood(
                    noise=train_std.pow(2).to(self._target)
                )
            )
        
        if self.fit_kernel:
            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            fit_gpytorch_mll(mll)
        else:
            if self.lengthscale is not None:
                if isinstance(self.lengthscale, list) and len(self.lengthscale) == 2:
                    # Set different lengthscales for each dimension
                    self.gpkernel.base_kernel.lengthscale = torch.tensor(self.lengthscale)
                else:
                    # Set same lengthscale for all dimensions
                    self.gpkernel.base_kernel.lengthscale = self.lengthscale
                    
            if self.outputscale is not None:
                self.gpkernel.outputscale = self.outputscale

    def acq(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate acquisition function value at point x"""
        if self.gp is None:
            raise ValueError("GP is not fitted yet.")
        acq = UncertaintyAcquisition(self.gp)
        return acq(x)

    def optimize_acq(self, q: int = 1, num_restarts: int = 20, raw_samples: int = 100) -> torch.Tensor:
        """Optimize acquisition function to find next point"""
        if self.gp is None:
            raise ValueError("GP is not fitted yet.")
        acq = UncertaintyAcquisition(self.gp)
        return optimize_acqf(acq, bounds=self._bounds, q=q, num_restarts=num_restarts, raw_samples=raw_samples)
    
    def find_candidate(self, q: int = 1, num_restarts: int = 20, raw_samples: int = 100) -> torch.Tensor:
        """Find next candidate point to measure"""
        return self.optimize_acq(q=q, num_restarts=num_restarts, raw_samples=raw_samples)[0]
    
    def plot_gp_2d(self,
                   test_x: torch.Tensor = None,
                   resolution: int = 50, 
                   figsize: tuple[int, int] = (14, 6)):
        """
        Plot the GP prediction and uncertainty on a 2D grid
        
        Args:
            test_x: Optional test points to plot - we ignore this.
            resolution: Number of points along each dimension
            figsize: Figure size (width, height)
        """
        if self.gp is None:
            raise ValueError("GP is not fitted yet.")
        
        # we ignore test_x.
            
        # Create a grid of test points
        x_range = torch.linspace(self._bounds[0, 0], self._bounds[1, 0], resolution)
        y_range = torch.linspace(self._bounds[0, 1], self._bounds[1, 1], resolution)
        grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
        test_x = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        
        # Get predictions from GP
        with torch.no_grad():
            posterior = self.gp.posterior(test_x)
            mean = posterior.mean.reshape(resolution, resolution)
            variance = posterior.variance.reshape(resolution, resolution)
            
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot mean prediction
        im1 = ax1.imshow(
            mean.t().numpy(),
            extent=[
                self._bounds[0, 0],
                self._bounds[1, 0],
                self._bounds[0, 1],
                self._bounds[1, 1]
            ],
            origin='lower',
            aspect='auto',
            cmap='viridis'
        )
        ax1.set_title('GP Mean Prediction')
        ax1.set_xlabel(self.axis_names[0])
        ax1.set_ylabel(self.axis_names[1])
        fig.colorbar(im1, ax=ax1)
        
        # Plot variance/uncertainty
        im2 = ax2.imshow(
            variance.t().numpy(),
            extent=[self._bounds[0, 0], self._bounds[1, 0], self._bounds[0, 1], self._bounds[1, 1]],
            origin='lower',
            aspect='auto',
            cmap='plasma'
        )
        ax2.set_title('GP Uncertainty')
        ax2.set_xlabel(self.axis_names[0])
        ax2.set_ylabel(self.axis_names[1])
        fig.colorbar(im2, ax=ax2)
        
        # Plot measured points
        ax1.scatter(self.train_x[:, 0], self.train_x[:, 1], c='r', marker='x', s=50, label='Observations')
        ax2.scatter(self.train_x[:, 0], self.train_x[:, 1], c='r', marker='x', s=50, label='Observations')
        
        ax1.legend()
        plt.tight_layout()
        return fig


class UncertaintyAcquisition(AnalyticAcquisitionFunction):
    """Acquisition function that maximizes uncertainty (variance)"""
    def __init__(self, model):
        super().__init__(model=model)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X)
        res = posterior.variance.squeeze((-1, -2))
        return res


if __name__ == "__main__":
    # Example usage
    active_measurement = Spatial2DimActiveMeasurement(
        initial_points=[[0, 0], [0, 1], [1, 0], [1, 1]],
        bounds=((-1, 1), (-1, 1)),
        fit_kernel=True,
        fit_likelihood=True,  # Demonstrate with likelihood fitting enabled
        max_num_points=30,
        lengthscale_prior_mean=[0.5, 0.5],  # Different priors for each dimension
        lengthscale_prior_std=[0.3, 0.3],
        outputscale_prior_mean=1.0,
        outputscale_prior_std=0.3,
        noise_prior_mean=0.01,  # Prior mean for noise (relatively small)
        noise_prior_std=0.5,    # Prior std allows some flexibility
        noise_bounds=(1e-6, 0.1),  # Reasonable bounds for noise
    )
    
    # Add initial points
    print("adding points")
    active_measurement.add_points(torch.tensor([0.0, 0.0]), torch.tensor([0.5]), torch.tensor([0.01]), update_gp=False)
    active_measurement.add_points(torch.tensor([0.0, 1.0]), torch.tensor([0.3]), torch.tensor([0.01]), update_gp=False)
    active_measurement.add_points(torch.tensor([1.0, 0.0]), torch.tensor([0.2]), torch.tensor([0.01]), update_gp=False)
    print("adding points with update_gp=True")
    active_measurement.add_points(torch.tensor([1.0, 1.0]), torch.tensor([0.8]), torch.tensor([0.01]), update_gp=True)
    
    print("finding candidate")
    # Find next candidate point
    candidate = active_measurement.find_candidate()
    print(f"Next point to measure: {candidate}")
    
    # Plot the GP
    active_measurement.plot_gp_2d()
    plt.savefig("gp_2d.png")
