import asyncio
import hydra
from omegaconf import DictConfig, OmegaConf
import importlib
from typing import Any, Type
from pathlib import Path
from datetime import datetime
import os
import shutil
from hydra.core.hydra_config import HydraConfig

from reflectgp.inference.preprocess_exp import StandardPreprocessing
from active_loop.xrr_config import XRRConfig


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point for running different active loops with different inference models
    
    Args:
        cfg: Hydra configuration
    """
    # Initialize save_dir before hydra can create any outputs
    save_dir = init_save_dir(cfg)
    # mkdir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the config to the custom save_dir
    config_save_path = save_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    # If Hydra created any output directory, copy its contents to our save_dir and remove it
    if HydraConfig.initialized() and HydraConfig.get().job.name != "":
        hydra_dir = Path(HydraConfig.get().runtime.output_dir)
        if hydra_dir.exists() and hydra_dir.is_dir() and hydra_dir != save_dir:
            # Copy any log files or other outputs to our save_dir
            for file_path in hydra_dir.glob("*"):
                if file_path.is_file():
                    shutil.copy2(file_path, save_dir)
            # No need to remove the directory as we're setting hydra.output_subdir to null in config
    
    print(f"Running with configuration:\n{OmegaConf.to_yaml(cfg)}")
    print(f"Configuration saved to {config_save_path}")
    
    # Create preprocessor from config
    if not hasattr(cfg, 'preprocessing') or not cfg.preprocessing:
        raise ValueError("Configuration must include 'preprocessing' section")
    
    prep_params = OmegaConf.to_container(cfg.preprocessing, resolve=True)
    preprocessor = StandardPreprocessing(**prep_params)
    
    # Import the active loop class
    active_loop_module = importlib.import_module(f"active_loop.{cfg.active_loop.module}")
    active_loop_class = getattr(active_loop_module, cfg.active_loop.class_name)
    
    # Import the inference model class
    inference_module = importlib.import_module(f"active_loop.{cfg.inference.module}")
    inference_class = getattr(inference_module, cfg.inference.class_name)
    
    # Instantiate the active measurement with its parameters
    active_measurement_module = importlib.import_module(f"active_loop.{cfg.active_measurement.module}")
    active_measurement_class = getattr(active_measurement_module, cfg.active_measurement.class_name)
    active_measurement_params = OmegaConf.to_container(cfg.active_measurement.params, resolve=True)
    active_measurement = active_measurement_class(**active_measurement_params)

    # Instantiate the inference model with its parameters
    inference_params = OmegaConf.to_container(cfg.inference.params, resolve=True)
    inference_params['preprocessor'] = preprocessor
    inference_params['save_dir'] = save_dir
    inference_params['active_measurement'] = active_measurement
    inference_model = inference_class(**inference_params)
    
    # Get XRR measurement config
    if not hasattr(cfg, 'xrr_config') or not cfg.xrr_config:
        raise ValueError("Configuration must include 'xrr_config' section")
    
    xrr_config = XRRConfig(**OmegaConf.to_container(cfg.xrr_config, resolve=True))
    
    # Instantiate the active loop with its parameters and the inference model
    loop_params = OmegaConf.to_container(cfg.active_loop.params, resolve=True)
    loop = active_loop_class(
        inference=inference_model, 
        xrr_config=xrr_config,
        **loop_params
    )
    
    # Run the active loop
    asyncio.run(run_with_loop(loop))


def init_save_dir(cfg: DictConfig) -> str:
    """Initialize the save directory for the active loop
    
    Args:
        cfg: Hydra configuration
    """
    root_dir = Path(__file__).parents[4]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{cfg.experiment.name}_{timestamp}"
    save_dir = root_dir / "results" / name
    print(f"Saving results to {save_dir}")
    return save_dir

async def run_with_loop(loop: Any) -> None:
    """Run the active loop
    
    Args:
        loop: The active loop instance
    """
    if await loop.connect():
        try:
            await loop.run_active_loop()
            print("Active loop complete")
        finally:
            await loop.disconnect()
    else:
        print("Failed to connect to servers")


if __name__ == "__main__":
    hydra_main()
