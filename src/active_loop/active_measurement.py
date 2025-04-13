
from typing import Optional

import torch
from torch import nn
import matplotlib.pyplot as plt


class ActiveMeasurement(nn.Module):
    def __init__(self,
                 initial_points: list[float],
                 bounds: tuple[float, float],
                 max_num_points: int,
                 ndim: int,
                 ):
        super().__init__()
        self.initial_points = initial_points
        self.x_min, self.x_max = bounds
        self.ndim = ndim
        self.max_num_points = max_num_points
        self.register_buffer(
            name="_bounds", tensor=torch.tensor([[bounds[0]], [bounds[1]]], dtype=torch.double)
        )
        self.register_buffer(
            name="_target", tensor=torch.tensor(1., dtype=torch.double)
        )
        self.register_buffer(
            name="train_x", tensor=torch.zeros(0, 1, dtype=torch.double)
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
