import torch
from torch import nn, Tensor
from numpy import ndarray

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
                 x: float,
                 intensity: ndarray,
                 scattering_angle: ndarray,
                 transmission: ndarray
                 ):
        res = {}
        res['raw_data'] = {
            'intensity': intensity,
            'scattering_angle': scattering_angle,
            'transmission': transmission,
        }
        res.update(self.preprocessor(intensity, scattering_angle, transmission))
        res['x'] = torch.tensor(x)
        self.data.append(res)
        self.logger.info(f"Added data point at x = {x}, total points: {len(self.data)}")
        self.process_new_data()

    def process_new_data(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_next_candidate(self) -> float:
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

                 ):
        super().__init__(
            preprocessor=preprocessor,
            active_measurement=active_measurement,
            log_level=log_level,
            log_file=log_file,
            save_dir=save_dir,
        )
        self.initial_points = active_measurement.initial_points
        self.logger.info(f"Initialized with {len(self.initial_points)} initial points: {self.initial_points}")

    def get_next_candidate(self) -> float:
        if len(self.data) < len(self.initial_points):
            next_point = self.initial_points[len(self.data)]
            self.logger.info(f"Using initial point: {next_point}")
            return next_point
        else:
            next_point = self.get_next_candidate_from_model()
            self.logger.info(f"Using model-predicted point: {next_point}")
            return next_point

    def get_next_candidate_from_model(self) -> float:
        candidate = self.active_measurement.find_candidate().flatten()[0].item()
        self.logger.info(f"Candidate: {candidate}")
        self.save_gp_plot()
        return candidate

    def save_gp_plot(self):
        self.active_measurement.plot_gp()
        path = self.save_dir / f"gp_plot_{len(self.data)}.png"
        plt.savefig(path)
        self.logger.info(f"GP plot saved to {path}")


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
        update_gp = len(self.data) > 1
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
    
    def save_last_results(self):
        if len(self.data) == 0:
            self.logger.warning("No data to save")
            return
        
        storage = self.data[-1]
        self.logger.info("Saving fit result")

        idx = len(self.data)
        name = f"fit_result_{self.config_name}_{idx}"
        res_path = self.save_dir / f"{name}.pt"
        torch.save({
            'data': storage,
            'active_measurement': self.active_measurement.state_dict(),
        }, res_path)
        self.logger.info(f"Fit result saved to {res_path}")
        return res_path
