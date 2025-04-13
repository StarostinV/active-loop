"""Dummy active learning loop that iterates over x values for measurements"""

import numpy as np
from typing import List, Optional, Dict, Any

from active_loop.ports import PUSH_SERVER_PORT, FETCH_SERVER_PORT

from active_loop.base_active_loop import BaseActiveLoop
from active_loop.consecutive_inference import ConsecutiveInference
from active_loop.xrr_config import XRRConfig


class ConsecutiveActiveLoop(BaseActiveLoop):
    """Active learning loop that uses consecutive inference.
    
    At the moment, only x optimization is supported.
    """
    
    def __init__(self,
                 inference: ConsecutiveInference,
                 host: str = '127.0.0.1',
                 push_port: Optional[int] = PUSH_SERVER_PORT,
                 fetch_port: Optional[int] = FETCH_SERVER_PORT,
                 log_level: str = 'INFO',
                 log_file: Optional[str] = None,
                 xrr_config: XRRConfig = None,
                 remove_background: bool = False,
                 ):
        """Initialize the dummy active learning loop
        
        Args:
            inference: The consecutive inference model to use
            host: The hostname to connect to
            push_port: Port for the push server
            fetch_port: Port for the fetch server
            log_level: Logging level
            log_file: Path to log file
            xrr_measurement: Configuration for XRR measurements
            remove_background: Whether to remove background from the data
        """
        super().__init__(
            host=host,
            push_port=push_port,
            fetch_port=fetch_port,
            log_level=log_level,
            log_file=log_file,
            xrr_config=xrr_config,
        )
        
        self.inference = inference
        self.remove_background = remove_background

    async def clean_up_before_loop(self):
        """Clean up before the active learning loop."""
        self.logger.info(f"Starting active loop.")
        
        # Reopen environment at the beginning of the loop
        self.logger.info("Reopening environment at beginning of active loop")
        await self.push_client.reopen_environment()
        
        # Clear scans and measurement queue at the beginning of the loop to free memory
        self.logger.info("Clearing scans at beginning of active loop")
        await self.fetch_client.clear_scans()
        self.logger.info("Clearing measurement queue at beginning of active loop")
        await self.push_client.clear_measurement_queue()
    
    async def run_active_loop(self) -> List[dict]:
        """Run the active learning loop.
        
        Returns:
            List of scan results
        """
        results = []
        
        await self.clean_up_before_loop()

        i = 0

        while not self.inference.is_complete():
            is_first_iteration = i == 0
            i += 1
            self.logger.info(f"Running active loop iteration {i}/{self.inference.max_num_points}")

            x = self.inference.get_next_candidate()
            self.logger.info(f"Next candidate: {x}")
            # Request measurement at this x position
            push_response = await self.push_client.request_measurement(
                x_positions=[x],
                clear_queue=is_first_iteration,  # Only clear queue on first iteration
                reopen_env=True,  # always reopen environment just in case
                **self.xrr_config.to_dict()
            )
            
            request_id = push_response.get('request_id')
            if 'error' in push_response:
                self.logger.error(f"Error requesting measurement: {push_response['error']}")
                continue
                
            self.logger.info(f"Measurement requested with ID {request_id}, waiting for results...")
            
            # Wait for the measurement to complete
            scan = await self._wait_for_scan()
            
            if scan:
                self.logger.info(f"Received scan for x = {x}")
                # Print some info about the scan
                scan_info = scan.get('info', {})
                self.logger.debug(f"Scan info: {scan_info}")
                
                # Print available streams and their lengths
                streams = scan.get('streams', {})
                for stream_name, stream_data in streams.items():
                    self.logger.info(f"Stream '{stream_name}' has {len(stream_data)} points")

                processed_curve = self.process_scan(scan)
                self.inference.add_data(x, **processed_curve)
                
                # Store result
                results.append({
                    'x_position': x,
                    'processed_curve': processed_curve,
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

        if self.remove_background:
            try:
                background = np.array(scan['streams']['background'])
                intensity = intensity - background
            except Exception as e:
                self.logger.warning(
                    f"Could not subtract background, using intensity as is: {e}"
                )

        return {
            'intensity': intensity,
            'scattering_angle': scattering_angle,
            'transmission': transmission,
        }
 