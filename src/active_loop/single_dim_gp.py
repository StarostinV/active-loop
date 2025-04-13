from typing import Optional

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.priors import NormalPrior, LogNormalPrior

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import AnalyticAcquisitionFunction, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from active_loop.active_measurement import ActiveMeasurement


class SingleDimActiveMeasurement(ActiveMeasurement):
    def __init__(self, 
                 initial_points: list[float],
                 gpkernel: Kernel = None,
                 bounds: tuple[float, float] = (0, 1),
                 param_idx: int = 0,
                 fit_kernel: bool = True,
                 max_num_points: int = 30,
                 lengthscale: float = None,
                 outputscale: float = None,
                 lengthscale_prior_mean: float = None,
                 lengthscale_prior_std: float = None,
                 outputscale_prior_mean: float = None,
                 outputscale_prior_std: float = None,
                 ):
        super().__init__(
            initial_points=initial_points,
            bounds=bounds,
            ndim=1,
            max_num_points=max_num_points,
        )
        if gpkernel is None:
            gpkernel = ScaleKernel(RBFKernel())
        self.gpkernel = gpkernel
        self.gp = None
        self.param_idx = param_idx
        self.fit_kernel = fit_kernel

        self.lengthscale = lengthscale
        self.outputscale = outputscale
        
        # Store prior parameters
        self.lengthscale_prior_mean = lengthscale_prior_mean
        self.lengthscale_prior_std = lengthscale_prior_std
        self.outputscale_prior_mean = outputscale_prior_mean
        self.outputscale_prior_std = outputscale_prior_std

    def preprocess_input(self, x, y, y_std):
        x = torch.atleast_1d(x.flatten())[:, None]
        y = torch.atleast_1d(y.flatten()[self.param_idx])[:, None]
        y_std = torch.atleast_1d(y_std.flatten()[self.param_idx])[:, None]
        return x, y, y_std
    
    def add_points(self, x: torch.Tensor, y: torch.Tensor , y_std: torch.Tensor, update_gp: bool = True):
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
            train_x: Training input points
            train_y: Training target values
            train_std: Optional point-dependent noise levels. 
        """
        # Configure kernel with priors if specified
        if self.lengthscale_prior_mean is not None and self.lengthscale_prior_std is not None:
            # Using LogNormal prior for lengthscale (must be positive)
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
                self.gpkernel.base_kernel.raw_lengthscale.data.fill_(self.lengthscale)
            if self.outputscale is not None:
                self.gpkernel.raw_outputscale.data.fill_(self.outputscale)

    def acq(self, x: torch.Tensor) -> torch.Tensor:
        if self.gp is None:
            raise ValueError("GP is not fitted yet.")
        acq = UncertaintyAcquisition(self.gp)
        return acq(x)

    def optimize_acq(self, q: int = 1, num_restarts: int = 20, raw_samples: int = 100) -> torch.Tensor:
        if self.gp is None:
            raise ValueError("GP is not fitted yet.")
        acq = UncertaintyAcquisition(self.gp)
        return optimize_acqf(acq, bounds=self.bounds, q=q, num_restarts=num_restarts, raw_samples=raw_samples)
    
    def find_candidate(self, q: int = 1, num_restarts: int = 20, raw_samples: int = 100) -> torch.Tensor:
        return self.optimize_acq(q=q, num_restarts=num_restarts, raw_samples=raw_samples)[0]


class UncertaintyAcquisition(AnalyticAcquisitionFunction):
    def __init__(self, model):
        super().__init__(model=model)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X)
        res = posterior.variance.squeeze((-1, -2))
        return res


if __name__ == "__main__":
    active_measurement = SingleDimActiveMeasurement(
        initial_points=[-24, 0, 24],
        bounds=[-24, 24],
        fit_kernel=True,  # Now we can use fit_kernel=True with priors
        max_num_points=30,
        lengthscale_prior_mean=1.0,
        lengthscale_prior_std=0.3,
        outputscale_prior_mean=1.0,
        outputscale_prior_std=0.3,
    )
    active_measurement.add_points(torch.tensor([-24]), torch.tensor([0.5]), torch.tensor([0.01]), update_gp=False)
    print("added points")
    active_measurement.add_points(torch.tensor([0]), torch.tensor([0.]), torch.tensor([0.01]), update_gp=True)
    print("added points")
    active_measurement.add_points(torch.tensor([24]), torch.tensor([-0.3]), torch.tensor([0.01]), update_gp=True)
    print("added points")
    print(active_measurement.train_x.shape)
    print(active_measurement.train_y.shape)
    print(active_measurement.train_y_std.shape)
    print(active_measurement)
    candidate = active_measurement.find_candidate()
    print(candidate)
