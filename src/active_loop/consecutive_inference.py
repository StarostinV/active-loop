import torch
from torch import nn, Tensor
from numpy import ndarray
import numpy as np
from reflectgp.inference.preprocess_exp import StandardPreprocessing
from reflectgp.config_utils import init_inference_model_from_config
from reflectgp.inference import MeasuredData
from reflectgp.transforms import Uniform2Normal

from active_loop.logging_utils import setup_logging
import matplotlib.pyplot as plt
from active_loop.active_measurement import ActiveMeasurement


class ConsecutiveInference(nn.Module):
    """Base class for consecutive inference.
    
    It receives data and fits a model to it.
    Then it predicts the next candidate to measure.
    """
    def __init__(self,
                 preprocessor: StandardPreprocessing,
                 active_measurement: ActiveMeasurement,
                 log_level: str = "INFO",
                 log_file: str = None,
                 save_dir: str = None,
                 ):
        super().__init__()

        self.preprocessor = preprocessor
        self.active_measurement = active_measurement
        self.data: list[dict] = []
        self.max_num_points = active_measurement.max_num_points
        # Setup logging
        self.logger = setup_logging(__name__, log_level, log_file)
        self.save_dir = save_dir


    def is_complete(self) -> bool:
        return len(self.data) >= self.max_num_points

    def add_data(self,
                 intensity: ndarray,
                 scattering_angle: ndarray,
                 transmission: ndarray,
                 x: ndarray,
                 ):
        res = {}
        res['raw_data'] = {
            'intensity': intensity,
            'scattering_angle': scattering_angle,
            'transmission': transmission,
            'x': x,
        }
        res['x'] = torch.tensor(x)
        self.data.append(res)
        preprocessed_data = self.preprocess_data(x, intensity, scattering_angle, transmission)
        res.update(preprocessed_data)
        self.logger.info(f"Added data point at x = {x}, total points: {len(self.data)}")
        self.process_new_data()

    def preprocess_data(self,
                        x: ndarray,
                        intensity: ndarray,
                        scattering_angle: ndarray,
                        transmission: ndarray
                        ) -> dict:
        return self.preprocessor(intensity, scattering_angle, transmission)

    def process_new_data(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_next_candidate(self) -> Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def clear_data(self):
        self.logger.info("Clearing all data")
        self.data = []


class BasicConsecutiveInference(ConsecutiveInference):
    """Basic consecutive inference that uses the data to
    fit a model and predict the next candidate.
    
    It receives a list of initial points to measure and only then
    it starts to make its own predictions.
    """
    def __init__(self,
                 preprocessor: StandardPreprocessing,
                 active_measurement: ActiveMeasurement,
                 log_level: str = "INFO",
                 log_file: str = None,
                 save_dir: str = None,
                 save_prefix: str = '',
                 ):
        super().__init__(
            preprocessor=preprocessor,
            active_measurement=active_measurement,
            log_level=log_level,
            log_file=log_file,
            save_dir=save_dir,
        )
        self.save_prefix = save_prefix
        self.initial_points = active_measurement.initial_points
        self.logger.info(f"Initialized with {len(self.initial_points)} initial points: {self.initial_points}")

    def get_next_candidate(self) -> Tensor:
        n_candidates = self.active_measurement.n_candidates
        
        # First use initial points, then switch to model predictions

        next_points = []

        while len(self.initial_points):
            next_points.append(self.initial_points.pop(0))

            if len(next_points) >= n_candidates:
                break
        
        if next_points:
            self.logger.info(f"Using initial point(s): {next_points}")

        if len(next_points) < n_candidates:
            if self.active_measurement.gp is not None:
                # Get point(s) from model
                additional_points = self.active_measurement.find_candidate(q=n_candidates - len(next_points))
                self.logger.info(f"Using model-predicted point(s): {additional_points}")
                self.save_gp_plot()

                if next_points:
                    next_points = torch.cat([torch.tensor(next_points), additional_points], 0)
                else:
                    next_points = additional_points
            else:
                self.logger.warning("No GP model found, using initial points only")
                next_points = torch.tensor(next_points)
        else:
            next_points = torch.tensor(next_points)

        xdim = self.active_measurement.xdim
        next_points = next_points.reshape(-1, xdim)
        self.logger.info(f"Next point(s): {next_points}")
        return next_points

    def save_gp_plot(self):
        self.active_measurement.plot_gp()
        path = self.save_dir / f"gp_plot_{len(self.data)}.png"
        plt.savefig(path)
        self.logger.info(f"GP plot saved to {path}")

    def save_last_results(self):
        if len(self.data) == 0:
            self.logger.warning("No data to save")
            return
        
        storage = self.data[-1]
        self.logger.info("Saving fit result")

        idx = len(self.data)
        name = f"fit_result_{self.save_prefix}_{idx}"
        res_path = self.save_dir / f"{name}.pt"
        torch.save({
            'data': storage,
            'active_measurement': self.active_measurement.state_dict(),
        }, res_path)
        self.logger.info(f"Fit result saved to {res_path}")
        return res_path



class DummyConsecutiveInferenceWithNPE(BasicConsecutiveInference):
    def __init__(self,
                 preprocessor: StandardPreprocessing,
                 active_measurement: ActiveMeasurement,
                 config_name: str = "gp-exp-thin-films-on-si",
                 device: str = 'cuda',
                 sigma: float = 0.2,
                 log_level: str = "INFO",
                 log_file: str = None,
                 save_dir: str = None,
                 phi: list = None,
                 ):
        super().__init__(
            preprocessor=preprocessor,
            active_measurement=active_measurement,
            log_level=log_level,
            log_file=log_file,
            save_dir=save_dir,
            save_prefix=config_name,
        )
        self.config_name = config_name
        self.logger.info(f"Initializing model with config: {self.config_name}")
        self.device = device
        self.model = self.init_model()
        self.constant_phi = self._init_constant_phi(phi).to(self.device)
        self.transform = Uniform2Normal(
            self.model.simulator.physical_model.theta_dim
        ).to(self.device)(self.constant_phi)
        self.sigma = sigma

    def init_model(self):
        model = init_inference_model_from_config(self.config_name)
        model.simulator.q_simulator.q_factor_mode = "one"
        model.to(self.device)
        self.logger.info(f"Model initialized on device: {self.device}. Model: \n{model}")
        return model

    def _init_constant_phi(self, phi=None) -> Tensor:
        self.logger.info("Initializing constant phi")
        if phi is None:
            res = self.model.sample_simulated_data(run_snis=False)
            phi = res.data.phi
        else:
            phi = torch.atleast_2d(torch.tensor(phi).to(self.device))
        return phi

    def convert_data_to_model_input(self, data: dict) -> MeasuredData:
        q = torch.tensor(data['q_values']).to(self.device).float()
        curve = torch.tensor(data['curve']).to(self.device).float()
        sigma = curve * self.sigma
        phi = self.constant_phi.to(self.device).float()

        self.logger.info(
            f"Shape of q: {q.shape}, shape of curve: {curve.shape}, shape of sigma: {sigma.shape}, shape of phi: {phi.shape}"
        )

        data = MeasuredData(q, curve, sigma, phi)
        return data
    
    def process_new_data(self):
        x, y, y_std = self.fit()
        update_gp = len(self.data) + x.shape[0] > 1
        self.active_measurement.add_points(x, y, y_std, update_gp=update_gp)
        self.save_last_results()
    
    def fit(self):
        if len(self.data) == 0:
            self.logger.warning("No data to fit")
            return
        
        self.logger.info("Fitting model to data")
        storage = self.data[-1]
        data = self.convert_data_to_model_input(storage)
        res = self.model(data)

        neff = res.importance_sampling.neff

        num_samples = max(min(500, int(neff)), 100)
        thetas = res.importance_sampling.sample(num_samples)[0].to(self.device)
        scaled_thetas = self.transform(thetas)
        scaled_means = scaled_thetas.mean(0)
        scaled_stds = scaled_thetas.std(0)
        means = thetas.mean(0)
        stds = thetas.std(0)

        if neff < 10:
            scaled_stds *= 10
        
        storage['mean'] = means.cpu()
        storage['std'] = stds.cpu()
        storage['scaled_mean'] = scaled_means.cpu()
        storage['scaled_std'] = scaled_stds.cpu()
        storage['thetas'] = thetas.cpu()
        storage['scaled_thetas'] = scaled_thetas.cpu()
        self.logger.info(f"Model fit complete, means: {means}, stds: {stds}")
        
        self.save_profile_plot(res)

        return storage['x'], storage['scaled_mean'], storage['scaled_std']
    
    def save_profile_plot(self, res):
        res.plot_sampled_profiles(show=False)
        path = self.save_dir / f"profiles_plot_{len(self.data)}.png"
        plt.savefig(path)
        self.logger.info(f"Profile plot saved to {path}")


class ConsecutiveInferenceWithXOMMap(BasicConsecutiveInference):
    def __init__(self,
                 preprocessor: StandardPreprocessing,
                 active_measurement: ActiveMeasurement,
                 log_level: str = "INFO",
                 log_file: str = None,
                 save_dir: str = None,
                 err_std: float = 0.2,
                 max_intensity: float = 10**10, 
                 save_prefix: str = '',
    ):
        super().__init__(
            preprocessor=preprocessor,
            active_measurement=active_measurement,
            log_level=log_level,
            log_file=log_file,
            save_dir=save_dir,
            save_prefix=save_prefix,
        )
        self.err_std = err_std
        self.max_intensity = max_intensity

    def add_data(self, 
                 x: np.ndarray,
                 intensity: ndarray,
                 scattering_angle: ndarray,
                 transmission: ndarray
                 ):
        res = {}
        x, intensity, intensity_err = self.preprocess_data(
            x, intensity, scattering_angle, transmission
        )
        res['raw_data'] = {
            'intensity': intensity,
            'scattering_angle': scattering_angle,
            'transmission': transmission,
        }
        res['x'] = torch.tensor(x)
        res['intensity'] = intensity
        res['intensity_err'] = intensity_err
        self.data.append(res)
        self.logger.info(f"Added data point at x = {x}, total points: {len(self.data)}")
        self.process_new_data()

    def preprocess_data(self,
                        x: np.ndarray,
                        intensity: ndarray,
                        scattering_angle: ndarray,
                        transmission: ndarray
                        ) -> dict:
        self.logger.info(f"x: {x}")
        self.logger.info(f"intensity: {intensity}")
        self.logger.info(f"scattering_angle: {scattering_angle}")
        self.logger.info(f"transmission: {transmission}")

        x_om = np.array([x, scattering_angle])

        indices = np.array(
            np.where(np.any(np.diff(x_om, axis=1) != 0, axis=0))[0].tolist() + [len(x) - 1]
        )

        x = np.stack([x[indices], scattering_angle[indices] / 2], -1)

        intensity = intensity[indices]
        scattering_angle = scattering_angle[indices]
        transmission = transmission[indices]

        self.logger.info(f"intensity: {intensity}")
        self.logger.info(f"scattering_angle: {scattering_angle}")
        self.logger.info(f"transmission: {transmission}")
        
        intensity = torch.as_tensor(intensity)
        transmission = torch.as_tensor(transmission)
        intensity = intensity / transmission
        intensity_err = torch.sqrt(intensity)
        scattering_angle = torch.as_tensor(scattering_angle)
        intensity = torch.clamp(intensity, min=1e-10)
        intensity_err = intensity_err / intensity / np.log(10)

        intensity = intensity / self.max_intensity
        intensity = torch.log10(intensity)

        intensity = intensity.unsqueeze(1)
        intensity_err = torch.clamp(intensity_err, min=1e-2, max=10).unsqueeze(1)

        self.logger.info(f"processed intensity: {intensity}")
        self.logger.info(f"processed intensity_err: {intensity_err}")
        self.logger.info(f"processed x: {x}")
        
        return x, intensity, intensity_err

    def process_new_data(self):
        data = self.data[-1]
        x, y, y_std = (
            data['x'],
            data['intensity'],
            data['intensity_err'],
        )
        update_gp = self.active_measurement.train_x.shape[0] + x.shape[0]> 1
        self.logger.info(
            f"x.shape: {x.shape}, y.shape: {y.shape}, y_std.shape: {y_std.shape}"
        )
        self.active_measurement.add_points(
            x, y, y_std,
            update_gp=update_gp
        )
        self.save_last_results()
 