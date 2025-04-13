
from typing import Optional

import torch
from torch import nn
import matplotlib.pyplot as plt


class ActiveMeasurement(nn.Module):
    def __init__(self,
                 initial_points: list[float],
                 bounds: tuple,
                 max_num_points: int,
                 ndim: int,
                 xdim: int = 1,
                 ):
        super().__init__()
        self.initial_points = initial_points
        if xdim == 1:
            self.x_min, self.x_max = bounds
        elif xdim == 2:
            self.x_min, self.x_max = bounds[0]
            self.y_min, self.y_max = bounds[1]
        else:
            raise ValueError(f"Invalid number of dimensions: {xdim}")

        self.ndim = ndim
        self.xdim = xdim
        self.max_num_points = max_num_points

        if xdim == 1:
            self.register_buffer(
                name="_bounds", tensor=torch.tensor([[bounds[0]], [bounds[1]]], dtype=torch.double)
            )
        else:
            assert xdim == 2
            self.register_buffer(
                name="_bounds", 
                tensor=torch.tensor([[bounds[0][0], bounds[1][0]], [bounds[0][1], bounds[1][1]]], 
                                    dtype=torch.double)
            )
        self.register_buffer(
            name="_target", tensor=torch.tensor(1., dtype=torch.double)
        )
        self.register_buffer(
            name="train_x", tensor=torch.zeros(0, self.xdim, dtype=torch.double)
        )
        self.register_buffer(
            name="train_y", tensor=torch.zeros(0, self.ndim, dtype=torch.double)
        )
        self.register_buffer(
            name="train_y_std", tensor=torch.zeros(0, self.ndim, dtype=torch.double)
        )

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds
    
    def add_points(self, x: torch.Tensor, y: torch.Tensor , y_std: torch.Tensor, update_gp: bool = True):
        raise NotImplementedError("Subclasses must implement this method")

    def find_candidate(self, q: int = 1, num_restarts: int = 20, raw_samples: int = 100) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def plot_gp(self, test_x: Optional[torch.Tensor] = None):
        if self.xdim == 1:
            self.plot_gp_1d(test_x)
        elif self.xdim == 2:
            self.plot_gp_2d(test_x)
        else:
            raise ValueError(f"GP plot not implemented for {self.xdim}D")

    def plot_gp_1d(self, test_x: Optional[torch.Tensor] = None):
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
            plt.legend()


    def plot_gp_2d(self, test_x: Optional[torch.Tensor] = None):
        raise NotImplementedError("Subclasses must implement this method")
