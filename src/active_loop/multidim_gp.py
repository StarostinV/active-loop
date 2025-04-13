from typing import Optional

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import  FixedNoiseGaussianLikelihood

from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


import torch

from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from botorch.models import KroneckerMultiTaskGP
from active_loop.active_measurement import ActiveMeasurement


class MultiDimActiveMeasurement(ActiveMeasurement):
    def __init__(self, 
                 initial_points: list[float],
                 gpkernel: Kernel = None,
                 bounds: tuple[float, float] = (0, 1),
                 param_indices: tuple[int, ...] = (0, 1),
                 max_num_points: int = 30,
                 ndim: int = 2,
                 ):
        super().__init__(
            initial_points=initial_points,
            bounds=bounds,
            ndim=ndim,
            max_num_points=max_num_points,
        )
        self.param_indices = param_indices
        self.gpkernel = gpkernel or RBFKernel()
        self.gp = None
        
    
    def add_points(self, x: torch.Tensor, y: torch.Tensor , y_std: torch.Tensor, update_gp: bool = True):
        self.train_x = torch.cat([self.train_x, x])
        self.train_y = torch.cat([self.train_y, y])
        self.train_y_std = torch.cat([self.train_y_std, y_std])
        if update_gp:
            self.fit(self.train_x, self.train_y, self.train_y_std)
    
    def fit(self,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            train_std: torch.Tensor,
            ):
        """
        Fit the GP model to the training data.
        
        Args:
            train_x: Training input points
            train_y: Training target values
            train_std: Optional point-dependent noise levels. If None, uses self.likelihood_std for all points.
        """
        batch_size = train_x.shape[0]
        assert train_y.shape[0] == batch_size
        assert train_std.shape[0] == batch_size
        assert train_x.shape[1] == 1
        assert train_y.shape[1] == self.ndim
        assert train_std.shape[1] == self.ndim

        self.gp = KroneckerMultiTaskGP(
            train_x,
            train_y,
            num_tasks=train_y.shape[-1],
        )
        self.gp.likelihood = FixedNoiseGaussianLikelihood(noise=train_std**2)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

    # def run_gp_fit(self):
    #     if self.gp is None:
    #         raise ValueError("GP is not fitted yet.")

    #     mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
    #     optim = torch.optim.Adam(self.gp.parameters(), lr=0.1)
    
    #     for _ in range(self.training_iterations):
    #         optim.zero_grad()
    #         output = self.gp(self.train_x)
    #         loss = -mll(output, self.train_y)
    #         loss.backward()
    #         optim.step()

    def acq(self, x: torch.Tensor) -> torch.Tensor:
        if self.gp is None:
            raise ValueError("GP is not fitted yet.")
        acq = MultiDimUncertaintyAcquisition(self.gp)
        return acq(x)

    def optimize_acq(self, q: int = 1, num_restarts: int = 20, raw_samples: int = 100) -> torch.Tensor:
        if self.gp is None:
            raise ValueError("GP is not fitted yet.")
        acq = MultiDimUncertaintyAcquisition(self.gp)
        return optimize_acqf(acq, bounds=self._bounds, q=q, num_restarts=num_restarts, raw_samples=raw_samples)
    
    def find_candidate(self, q: int = 1, num_restarts: int = 20, raw_samples: int = 100) -> torch.Tensor:
        return self.optimize_acq(q=q, num_restarts=num_restarts, raw_samples=raw_samples)[0]

    def plot_gp(self, test_x: Optional[torch.Tensor] = None):
        if test_x is None:
            test_x = torch.linspace(self.x_min, self.x_max, 100).unsqueeze(-1)
        with torch.no_grad():
            posterior = self.gp.posterior(test_x)
            mean = posterior.mean
            lower, upper = posterior.mvn.confidence_region()

            plt.figure(figsize=(10, 6))
            plt.plot(test_x.numpy(), mean.numpy(), 'b-', label='GP mean')
            plt.fill_between(test_x.squeeze().numpy(), 
                           lower.numpy(), upper.numpy(), 
                           alpha=0.2, color='b', label='GP uncertainty')
            plt.scatter(self.train_x.numpy(), self.train_y.numpy(), 
                       c='r', marker='x', label='Measurements')



class MultiDimUncertaintyAcquisition(AnalyticAcquisitionFunction):
    def __init__(self, model):
        super().__init__(model=model)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X)
        res = posterior.variance.squeeze((-1, -2))
        return res


if __name__ == "__main__":
    kernel = RBFKernel()
    measurement = MultiDimActiveMeasurement(
        gpkernel=kernel, 
        bounds=(0.0, 1.0),
        param_indices=(0, 1),
        initial_points=[0.0, 0.0],
        max_num_points=30,
        ndim=2,
    )

    # Get initial points
    measurement.add_points(
        torch.tensor([[0.0,]]), 
        torch.tensor([[0.0, 0.0]]), 
        torch.tensor([[0.01, 0.01]]), update_gp=False)
    measurement.add_points(
        torch.tensor([[1.0,]]), 
        torch.tensor([[1.0, 1.0]]), 
        torch.tensor([[0.01, 0.01]]), update_gp=True)
