"""Server with the fetch loop"""

import asyncio
import json
import logging
import os
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from blissdata.redis_engine.store import DataStore
from blissdata.redis_engine.exceptions import NoScanAvailable

from active_loop.ports import FETCH_SERVER_PORT, PROD_REDIS_CONFIG, TEST_REDIS_CONFIG
from active_loop.network_utils import read_json_message
from active_loop.logging_utils import setup_logging


@dataclass
class ScanData:
    """Data structure to hold scan information and streams"""
    info: Dict[str, Any]
    streams: Dict[str, List]
    timestamp: float


class FetchServer:
    """Server with the fetch loop"""

    def __init__(self, 
                 host: str = '127.0.0.1', 
                 port: int = FETCH_SERVER_PORT, 
                 use_prod: bool = True, 
                 log_level: str = 'INFO',
                 intensity_key: str = "p100k_roi2",
                 transmission_key: str = "transmission", 
                 tt_key: str = "tt",
                 background_key: str = "p100k_roi3",
                 fetch_interval: float = 1.0,
                 save_to_file: bool = False,
                 log_file: Optional[str] = None,
                 timeout: float = 1.0,
                 ):
        self.host = host
        self.port = port
        self.use_prod = use_prod
        self.log_level = log_level
        # Keys for extracting data
        self.background_key = background_key
        self.intensity_key = intensity_key
        self.transmission_key = transmission_key
        self.tt_key = tt_key
        self.fetch_interval = fetch_interval
        self.save_to_file = save_to_file
        self.timeout = timeout
        # Redis configuration
        redis_config = PROD_REDIS_CONFIG if use_prod else TEST_REDIS_CONFIG
        self.redis_url = "redis://haspp08:6380"  # Default Redis URL
        
        # Data storage
        self.scans = []
        self.fetch_task = None
        self.running = False

        if log_file is None:
            log_file = f"fetch_server_{int(time.time())}.log"
        
        # set up logging
        self.logger = setup_logging(__name__, log_level, log_file)

    async def start(self):
        """Start the server and the fetch loop"""
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        self.running = True
        self.fetch_task = asyncio.create_task(self.fetch_loop())
        self.logger.info(f"Fetch server started on {self.host}:{self.port}")
        async with self.server:
            await self.server.serve_forever()

    async def stop(self):
        """Stop the server and fetch loop"""
        self.running = False
        if self.fetch_task:
            self.fetch_task.cancel()
            try:
                await self.fetch_task
            except asyncio.CancelledError:
                pass
        
        self.server.close()
        await self.server.wait_closed()
        self.logger.info("Fetch server stopped")

    async def fetch_loop(self):
        """Loop to continuously fetch new scans"""
        data_store = DataStore(self.redis_url)
        last_key = None
        data_is_fetched = False
        
        while self.running:
            await asyncio.sleep(self.fetch_interval)
            try:
                try:
                    self.logger.debug("Fetching next scan")
                    timestamp, key = data_store.get_next_scan(timeout=self.timeout)
                    self.logger.info(f"Identified next scan: {key}")
                except NoScanAvailable:
                    self.logger.debug("No new scan available, fetching last scan")
                    timestamp, key = data_store.get_last_scan()
                    self.logger.debug(f"Fetched last scan: {key}")
                
                # Skip if we've already seen this scan
                if key == last_key:
                    self.logger.debug("Same scan as before")
                    if data_is_fetched:
                        self.logger.debug("Data is fetched, waiting for new scan")
                        await asyncio.sleep(self.fetch_interval)
                        continue
                    else:
                        self.logger.debug("No data is fetched yet, try again.")
                else:
                    # new key, reset flag
                    self.logger.info(f"Fetched new scan: {key}")
                    data_is_fetched = False

                last_key = key
                scan = data_store.load_scan(key)
                self.logger.debug(f"Scan: {scan}")
                
                # Extract the streams we care about
                scan_data = ScanData(
                    info=scan.info,
                    streams={
                        "intensity": scan.streams[self.intensity_key][:].tolist() if self.intensity_key in scan.streams else [],
                        "transmission": scan.streams[self.transmission_key][:].tolist() if self.transmission_key in scan.streams else [],
                        "scattering_angle": scan.streams[self.tt_key][:].tolist() if self.tt_key in scan.streams else [],
                        "background": scan.streams[self.background_key][:].tolist() if self.background_key in scan.streams else [],
                    },
                    timestamp=timestamp
                )

                if len(scan_data.streams["intensity"]) == 0:
                    self.logger.debug("No data is fetched yet, try again.")
                    data_is_fetched = False
                else:
                    data_is_fetched = True
                    self.scans.append(scan_data)
                    self.logger.info(f"Added scan. Total scans: {len(self.scans)}")
                    if self.save_to_file:
                        self.save_scan_to_file(scan_data)
                        self.logger.info(f"Saved scan to file.")

            except Exception as e:
                if "Negative index have no meaning before stream is sealed" in str(e):
                    self.logger.debug(f"Expected error: {e}")
                else:
                    self.logger.error(f"Error fetching scan: {e}")
            
            await asyncio.sleep(self.fetch_interval)

    def save_scan_to_file(self, scan_data: ScanData):
        """Save the scan data to a file"""
        try:
            # Convert ScanData to a serializable dict
            scan_dict = {
                'info': scan_data.info,
                'streams': scan_data.streams,
                'timestamp': scan_data.timestamp
            }
            
            # Create a timestamped filename
            filename = f"scan_{len(self.scans)}_{int(time.time())}.json"
            self.logger.info(f"Saving scan to file: {filename}")
            
            with open(filename, "w") as f:
                json.dump(scan_dict, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving scan to file: {e}")

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a client connection"""
        addr = writer.get_extra_info('peername')
        self.logger.info(f"Client connected: {addr}")
        
        try:
            while True:
                try:
                    # Use utility function to read full JSON message
                    request = await read_json_message(reader)
                    
                    command = request.get('command')
                    response = {}
                    
                    if command == 'get_num_new_scans':
                        response = {'count': len(self.scans)}
                    
                    elif command == 'get_last_scan':
                        if self.scans:
                            response = {'scan': self._serialize_scan(self.scans[-1])}
                        else:
                            response = {'error': 'No scans available'}
                    
                    elif command == 'get_scan_by_idx':
                        idx = request.get('idx', 0)
                        if 0 <= idx < len(self.scans):
                            response = {'scan': self._serialize_scan(self.scans[idx])}
                        else:
                            response = {'error': f'Invalid scan index: {idx}'}
                    
                    elif command == 'get_all_scans':
                        response = {'scans': [self._serialize_scan(scan) for scan in self.scans]}
                    
                    elif command == 'clear_scans':
                        self.clear_scans()
                        response = {'success': True, 'message': 'All scans cleared'}
                    
                    else:
                        response = {'error': f'Unknown command: {command}'}
                    
                    # Send response
                    writer.write(json.dumps(response).encode())
                    await writer.drain()
                    
                except (json.JSONDecodeError, ConnectionError, asyncio.IncompleteReadError):
                    # Handle disconnection or bad data gracefully
                    self.logger.info(f"Client disconnected: {addr}")
                    break
                except Exception as e:
                    self.logger.error(f"Error processing request: {e}")
                    try:
                        writer.write(json.dumps({'error': str(e)}).encode())
                        await writer.drain()
                    except (ConnectionError, BrokenPipeError):
                        self.logger.info(f"Connection lost while sending error response: {addr}")
                        break
        
        except Exception as e:
            self.logger.error(f"Client connection error: {e}")
        finally:
            try:
                writer.close()
                # Only wait for close if the transport is not already closed
                if not writer.is_closing():
                    await writer.wait_closed()
            except Exception as e:
                self.logger.debug(f"Error while closing connection: {e}")
            
            self.logger.info(f"Client disconnected: {addr}")
    
    def _serialize_scan(self, scan: ScanData) -> dict:
        """Serialize a scan for JSON transmission"""
        return {
            'info': scan.info,
            'streams': scan.streams,
            'timestamp': scan.timestamp
        }

    def clear_scans(self):
        """Clear all scans to free up memory"""
        count = len(self.scans)
        self.scans = []
        self.logger.info(f"Cleared {count} scans from memory")


if __name__ == "__main__":
    server = FetchServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        pass


def main():
    """Entry point for the fetch server command-line tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Server for scan data")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=FETCH_SERVER_PORT, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                      help="Logging level")
    parser.add_argument("--save-to-file", action="store_true", help="Save the scan data to a file")
    parser.add_argument("--log-file", default=None, help="File to save logs to")
    args = parser.parse_args()
    
    server = FetchServer(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        save_to_file=args.save_to_file,
        log_file=args.log_file
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nShutting down server...")

