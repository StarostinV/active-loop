"""Dummy active learning loop that iterates over x values for measurements"""

import asyncio
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Optional, Dict, Any

from active_loop.push_client import PushClient
from active_loop.fetch_client import FetchClient
from active_loop.ports import PUSH_SERVER_PORT, FETCH_SERVER_PORT
from active_loop.xrr_config import XRRConfig
from active_loop.logging_utils import setup_logging


class BaseActiveLoop:
    """Base class for active learning loops"""
    
    def __init__(self,
                 host: str = '127.0.0.1',
                 push_port: Optional[int] = PUSH_SERVER_PORT,
                 fetch_port: Optional[int] = FETCH_SERVER_PORT,
                 log_level: str = 'INFO',
                 log_file: Optional[str] = None,
                 xrr_config: XRRConfig = None,
                 max_attempts: int = 1000,
                 wait_time: float = 0.5,
                 ):
        """Initialize the dummy active learning loop
        
        Args:
            host: The hostname to connect to
            push_port: Port for the push server
            fetch_port: Port for the fetch server
            log_level: Logging level
            log_file: Path to log file
            xrr_measurement: Configuration for XRR measurements
        """
        # Set up logging
        self.logger = setup_logging(__name__, log_level, log_file)
        
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
        
        self.xrr_config = xrr_config or XRRConfig()
        self.max_attempts = max_attempts
        self.wait_time = wait_time
    
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
        
        This is a base implementation that should be overridden by subclasses.
        
        Returns:
            List of scan results
        """
        raise NotImplementedError("Subclasses must implement run_active_loop")
    
    async def _wait_for_scan(self) -> Optional[dict]:
        """Wait for a new scan to appear
        
        Args:
            max_attempts: Maximum number of attempts to check for new scans
            wait_time: Time to wait between attempts in seconds
            
        Returns:
            The scan data or None if no new scan appeared
        """
        wait_time = self.wait_time
        max_attempts = self.max_attempts
        
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
