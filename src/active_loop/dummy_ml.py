"""Dummy active learning loop that iterates over x values for measurements"""

import asyncio
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Optional

from active_loop.push_client import PushClient
from active_loop.fetch_client import FetchClient
from active_loop.ports import PUSH_SERVER_PORT, FETCH_SERVER_PORT

from reflectgp.inference.preprocess_exp import StandardPreprocessing

class DummyActiveLoop:
    """A simple demonstration of an active learning loop"""
    
    def __init__(self,
                 host: str = '127.0.0.1',
                 push_port: Optional[int] = PUSH_SERVER_PORT,
                 fetch_port: Optional[int] = FETCH_SERVER_PORT,
                 log_level: str = 'INFO',
                 log_file: Optional[str] = None):
        """Initialize the dummy active learning loop
        
        Args:
            host: The hostname to connect to
            push_port: Port for the push server
            fetch_port: Port for the fetch server
            log_level: Logging level
            log_file: Path to log file
        """
        # Set up logging
        self._setup_logging(log_level, log_file)
        
        # Create clients
        self.push_client = PushClient(
            host=host,
            port=push_port,
            log_level=log_level,
            log_file=log_file
        )
        
        self.fetch_client = FetchClient(
            host=host, 
            port=fetch_port,
            log_level=log_level,
            log_file=log_file
        )

        self.preprocessor = StandardPreprocessing(
            wavelength=0.6888,
            beam_width=0.1,
            sample_length=12,
            beam_shape="gauss",
            normalize_mode="max",
        )
    
    def _setup_logging(self, log_level: str, log_file: Optional[str] = None):
        """Set up logging configuration"""
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(numeric_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if log_file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to file: {log_file}")
    
    async def connect(self) -> bool:
        """Connect to both push and fetch servers"""
        push_connected = await self.push_client.connect()
        fetch_connected = await self.fetch_client.connect()
        
        if not push_connected:
            self.logger.error("Failed to connect to push server")
            return False
        
        if not fetch_connected:
            self.logger.error("Failed to connect to fetch server")
            await self.push_client.disconnect()
            return False
        
        return True
    
    async def disconnect(self):
        """Disconnect from both servers"""
        await self.push_client.disconnect()
        await self.fetch_client.disconnect()
    
    async def _wait_for_scan(self, max_attempts: int = 1000, wait_time: float = 0.5) -> Optional[dict]:
        """Wait for a new scan to appear
        
        Args:
            max_attempts: Maximum number of attempts to check for new scans
            wait_time: Time to wait between attempts in seconds
            
        Returns:
            The scan data or None if no new scan appeared
        """
        self.logger.info(f"Waiting for scan, checking every {wait_time}s, max {max_attempts} attempts")
        
        # Get initial count of scans
        initial_count = await self.fetch_client.get_num_new_scans()
        self.logger.info(f"Initial scan count: {initial_count}")
        
        # Wait for new scan to appear
        for attempt in range(max_attempts):
            await asyncio.sleep(wait_time)
            current_count = await self.fetch_client.get_num_new_scans()
            
            if current_count > initial_count:
                self.logger.info(f"New scan detected on attempt {attempt+1}")
                scan = await self.fetch_client.get_last_scan()
                return scan
            
            self.logger.debug(f"No new scan yet (attempt {attempt+1}/{max_attempts})")
        
        self.logger.warning(f"No new scan appeared after {max_attempts} attempts")
        return None
    
    async def run_active_loop(self, 
                             x_values: List[float],
                             tt_max: float = 1.0,
                             num_points: int = 64,
                             max_wait_attempts: int = 1000,
                             wait_time: float = 0.5) -> List[dict]:
        """Run the active learning loop over the given x values
        
        Args:
            x_values: List of x positions to measure
            tt_max: Maximum tt angle
            num_points: Number of measurement points
            max_wait_attempts: Maximum attempts to wait for a scan
            wait_time: Time to wait between attempts in seconds
            
        Returns:
            List of scan results
        """
        results = []
        
        self.logger.info(f"Starting active loop with {len(x_values)} x values: {x_values}")
        
        # Reopen environment at the beginning of the loop
        self.logger.info("Reopening environment at beginning of active loop")
        await self.push_client.reopen_environment()
        
        # Clear scans and measurement queue at the beginning of the loop to free memory
        self.logger.info("Clearing scans at beginning of active loop")
        await self.fetch_client.clear_scans()
        self.logger.info("Clearing measurement queue at beginning of active loop")
        await self.push_client.clear_measurement_queue()
        
        for i, x in enumerate(x_values):
            self.logger.info(f"Iteration {i+1}/{len(x_values)}: x = {x}")
            
            # Request measurement at this x position
            push_response = await self.push_client.request_measurement(
                x_positions=[x],
                tt_max=tt_max,
                num_points=num_points,
                clear_queue=(i == 0),  # Only clear queue on first iteration
                close_environment=(i == len(x_values) - 1)  # Only close on last iteration
            )
            
            request_id = push_response.get('request_id')
            if 'error' in push_response:
                self.logger.error(f"Error requesting measurement: {push_response['error']}")
                continue
                
            self.logger.info(f"Measurement requested with ID {request_id}, waiting for results...")
            
            # Wait for the measurement to complete
            scan = await self._wait_for_scan(max_attempts=max_wait_attempts, wait_time=wait_time)
            
            if scan:
                self.logger.info(f"Received scan for x = {x}")
                # Print some info about the scan
                scan_info = scan.get('info', {})
                self.logger.info(f"Scan info: {scan_info}")
                
                # Print available streams and their lengths
                streams = scan.get('streams', {})
                for stream_name, stream_data in streams.items():
                    self.logger.info(f"Stream '{stream_name}' has {len(stream_data)} points")

                processed_curve = self.process_scan(scan)
                
                # Store result
                results.append({
                    'x_position': x,
                    'curve': processed_curve,
                })
            else:
                self.logger.warning(f"No scan received for x = {x} after waiting")
        
        # Clear scans and measurement queue at the end of the loop to free memory
        self.logger.info("Clearing scans at end of active loop")
        await self.fetch_client.clear_scans()
        self.logger.info("Clearing measurement queue at end of active loop")
        await self.push_client.clear_measurement_queue()
        
        self.logger.info(f"Active loop complete, collected {len(results)} scans")
        return results
    
    def process_scan(self, scan: dict) -> dict:
        """Process a scan and return a dictionary of processed data"""
        # Get intensity and tt data
        intensity = np.array(scan['streams']['intensity'])
        scattering_angle = np.array(scan['streams']['scattering_angle'])
        transmission = np.array(scan['streams']['transmission'])

        # self.logger.info(f"Intensity min: {np.min(intensity)}, max: {np.max(intensity)}")
        # self.logger.info(f"TT min: {np.min(scattering_angle)}, max: {np.max(scattering_angle)}")
        # self.logger.info(f"Transmission min: {np.min(transmission)}, max: {np.max(transmission)}")

        # remove non-positive scattering angles
        mask = scattering_angle > 0
        intensity = intensity[mask]
        scattering_angle = scattering_angle[mask]
        transmission = transmission[mask]

        # save data to file
        np.savez('data.npz', intensity=intensity, scattering_angle=scattering_angle, transmission=transmission)

        res = self.preprocessor(
            intensity=np.array(intensity),
            scattering_angle=np.array(scattering_angle),
            attenuation=1 / np.array(transmission)
        )

        self.logger.info(f"Processed scan with {len(res['curve'])} points")
        self.save_curve_plot(res)
        return res
    
    def save_curve_plot(self, res):
        """Save a plot of the curve to a file"""
        import matplotlib.pyplot as plt
        import numpy as np

        q = res['q_values']
        intensity = res['curve']

        plt.figure(figsize=(10, 6))
        plt.semilogy(q, intensity)
        plt.xlabel('Q')
        plt.ylabel('Intensity')
        plt.title('Curve')
        plt.savefig('current_curve.png')
        plt.close()
        


async def main_async():
    parser = argparse.ArgumentParser(description="Dummy Active Learning Loop")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--push-port", type=int, default=PUSH_SERVER_PORT, help="Push server port")
    parser.add_argument("--fetch-port", type=int, default=FETCH_SERVER_PORT, help="Fetch server port")
    parser.add_argument("--log-level", default="INFO", 
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                      help="Logging level")
    parser.add_argument("--log-file", default=None, help="File to save logs to")
    
    # Measurement parameters
    parser.add_argument("--x-min", type=float, default=-10.0, help="Minimum x value")
    parser.add_argument("--x-max", type=float, default=10.0, help="Maximum x value")
    parser.add_argument("--x-step", type=float, default=1.0, help="Step size for x values")
    parser.add_argument("--tt-max", type=float, default=1.0, help="Maximum tt angle")
    parser.add_argument("--num-points", type=int, default=64, help="Number of measurement points")
    parser.add_argument("--wait-time", type=float, default=0.5, 
                      help="Wait time between scan checks (seconds)")
    parser.add_argument("--max-wait", type=int, default=1000, 
                      help="Maximum number of attempts to wait for scan")
    
    args = parser.parse_args()
    
    # Generate x values
    x_values = np.arange(args.x_min, args.x_max + args.x_step/2, args.x_step).tolist()
    
    print(f"Starting dummy active learning loop with {len(x_values)} x values")
    print(f"X values: {x_values}")
    
    loop = DummyActiveLoop(
        host=args.host,
        push_port=args.push_port,
        fetch_port=args.fetch_port,
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    if await loop.connect():
        try:
            results = await loop.run_active_loop(
                x_values=x_values,
                tt_max=args.tt_max,
                num_points=args.num_points,
                max_wait_attempts=args.max_wait,
                wait_time=args.wait_time
            )
            
            print(f"Active loop complete, collected {len(results)} scans")
            
        finally:
            await loop.disconnect()
    else:
        print("Failed to connect to servers")


def main():
    """Entry point for the dummy active learning loop"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
