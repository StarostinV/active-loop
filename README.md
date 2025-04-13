# Active Loop

A framework for running active learning loops for X-ray scattering experiments.

## Installation

```bash
pip install -e .
```

## Usage

### Using Hydra Configuration

The active loop framework supports Hydra for configuration management, making it easy to switch between different active loops and inference models.

#### Basic Command

```bash
active-run
```

This will run the default configuration (ConsecutiveActiveLoop with NPE inference).

#### Overriding Configuration Values

You can override any configuration value from the command line:

```bash
active-run active_loop=dummy_loop
```

This will use the DummyActiveLoop instead of the ConsecutiveActiveLoop.

#### Specifying Inference Model

```bash
active-run inference=npe_inference
```

#### Changing Connection Parameters

```bash
active-run connection.host=192.168.1.100 connection.push_port=9002
```

#### Changing Measurement Parameters

```bash
active-run measurement.tt_max=2.0 measurement.num_points=128
```

#### Full Example

```bash
active-run active_loop=consecutive_loop inference=npe_inference \
  connection.host=192.168.1.100 \
  connection.log_level=DEBUG \
  measurement.tt_max=2.0 \
  measurement.num_points=128 \
  inference.params.initial_points=[-5,-2.5,0,2.5,5] \
  inference.params.device=cpu
```

### Using the CLI Directly

If you prefer to use command-line arguments directly without Hydra:

```bash
# For the dummy active loop
active-dummy-ml --host 127.0.0.1 --x-min -5 --x-max 5 --x-step 0.5

# For the consecutive active loop
active-consecutive --host 127.0.0.1 --initial-points -5 -2.5 0 2.5 5 --tt-max 2.0
```

## Adding New Active Loops and Inference Models

1. Create a new active loop class in `active_loop/your_active_loop.py`
2. Create a new inference model class in `active_loop/your_inference.py`
3. Add configuration files:
   - `active_loop/src/conf/active_loop/your_active_loop.yaml`
   - `active_loop/src/conf/inference/your_inference.yaml`

## Configuration Files

The configuration files are located in the following directories:

- `active_loop/src/conf/config.yaml` - Main configuration
- `active_loop/src/conf/active_loop/*.yaml` - Active loop configurations
- `active_loop/src/conf/inference/*.yaml` - Inference model configurations 