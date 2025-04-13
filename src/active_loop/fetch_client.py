"""Client for talking to the fetch loop for control and fetching data"""

import asyncio
import json
import logging
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from active_loop.ports import FETCH_SERVER_PORT
from active_loop.network_utils import read_json_message
from active_loop.logging_utils import setup_logging


class FetchClient:
    """Client for interacting with the fetch server"""
    
    def __init__(self, 
                 host: str = '127.0.0.1', 
                 port: int = FETCH_SERVER_PORT,
                 log_level: str = 'INFO',
                 log_file: Optional[str] = None):
        self.host = host
        self.port = port
        
        # Setup logging
        self.logger = setup_logging(__name__, log_level, log_file)
        
        self._reader = None
        self._writer = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to the fetch server"""
        try:
            self.logger.info(f"Connecting to fetch server at {self.host}:{self.port}...")
            self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
            self._connected = True
            self.logger.info(f"Connected to fetch server.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to fetch server: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the fetch server"""
        if self._connected and self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._connected = False
            self.logger.info("Disconnected from fetch server")
    
    async def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to the server and get the response"""
        if not self._connected:
            raise ConnectionError("Not connected to fetch server")
        
        try:
            # Send command
            command_str = json.dumps(command)
            self.logger.debug(f"Sending command: {command_str}")
            self._writer.write(command_str.encode())
            await self._writer.drain()
            
            # Get response using the shared utility function
            response = await read_json_message(self._reader)
            
            self.logger.debug(f"Received response: {response.keys()}")
            return response
        
        except Exception as e:
            self.logger.error(f"Error communicating with server: {e}")
            raise
    
    async def get_num_new_scans(self) -> int:
        """Get the number of available scans"""
        self.logger.debug("Getting number of new scans")
        response = await self._send_command({'command': 'get_num_new_scans'})
        count = response.get('count', 0)
        self.logger.debug(f"Found {count} scans")
        return count
    
    async def get_last_scan(self) -> Optional[Dict[str, Any]]:
        """Get the most recent scan"""
        self.logger.info("Getting last scan")
        response = await self._send_command({'command': 'get_last_scan'})
        if 'error' in response:
            self.logger.warning(f"Error getting last scan: {response['error']}")
            return None
        self.logger.info("Successfully fetched last scan")
        return response.get('scan')
    
    async def get_scan_by_idx(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a specific scan by index"""
        self.logger.info(f"Getting scan at index {idx}")
        response = await self._send_command({'command': 'get_scan_by_idx', 'idx': idx})
        if 'error' in response:
            self.logger.warning(f"Error getting scan {idx}: {response['error']}")
            return None
        self.logger.info(f"Successfully fetched scan at index {idx}")
        return response.get('scan')
    
    async def get_all_scans(self) -> List[Dict[str, Any]]:
        """Get all available scans"""
        self.logger.info("Getting all scans")
        response = await self._send_command({'command': 'get_all_scans'})
        scans = response.get('scans', [])
        self.logger.info(f"Successfully fetched {len(scans)} scans")
        return scans
    
    async def clear_scans(self) -> bool:
        """Clear all scans from the server to free up memory"""
        self.logger.info("Clearing all scans from server")
        response = await self._send_command({'command': 'clear_scans'})
        success = response.get('success', False)
        if success:
            self.logger.info("Successfully cleared all scans")
        else:
            self.logger.warning(f"Failed to clear scans: {response.get('error', 'Unknown error')}")
        return success
    
    def _convert_streams_to_numpy(self, scan: Dict[str, Any]) -> Dict[str, Any]:
        """Convert stream lists to numpy arrays"""
        if not scan or 'streams' not in scan:
            return scan
            
        result = dict(scan)
        result['streams'] = {
            key: np.array(values) for key, values in scan['streams'].items()
        }
        return result


# Example usage
async def example_usage():
    client = FetchClient()
    if await client.connect():
        try:
            # Get scan count
            count = await client.get_num_new_scans()
            print(f"Available scans: {count}")
            
            # Get latest scan
            scan = await client.get_last_scan()
            if scan:
                print(f"Latest scan info: {scan['info']}")
                
            # Get all scans
            all_scans = await client.get_all_scans()
            print(f"Total scans retrieved: {len(all_scans)}")
            
            # Clear all scans
            await client.clear_scans()
            
        finally:
            await client.disconnect()


if __name__ == "__main__":
    asyncio.run(example_usage())


def main():
    """Entry point for the fetch client command-line tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Client for scan data")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=FETCH_SERVER_PORT, help="Server port")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                      help="Logging level")
    parser.add_argument("--log-file", default=None, help="File to save logs to")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)
    
    # Count command
    subparsers.add_parser("count", help="Get number of available scans")
    
    # Last scan command
    subparsers.add_parser("last", help="Get the last scan")
    
    # Get scan by index command
    get_parser = subparsers.add_parser("get", help="Get scan by index")
    get_parser.add_argument("index", type=int, help="Scan index")
    
    # List all scans command
    subparsers.add_parser("list", help="List all available scans")
    
    # Clear all scans command
    subparsers.add_parser("clear", help="Clear all scans from server memory")
    
    args = parser.parse_args()
    
    async def run_command():
        client = FetchClient(
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            log_file=args.log_file
        )
        
        if not await client.connect():
            print(f"Failed to connect to server at {args.host}:{args.port}")
            return
        
        try:
            if args.command == "count":
                count = await client.get_num_new_scans()
                print(f"Number of available scans: {count}")
                
            elif args.command == "last":
                scan = await client.get_last_scan()
                if scan:
                    print(f"Last scan info: {json.dumps(scan['info'], indent=2)}")
                    print(f"Available streams: {list(scan['streams'].keys())}")
                else:
                    print("No scans available")
                    
            elif args.command == "get":
                scan = await client.get_scan_by_idx(args.index)
                if scan:
                    print(f"Scan {args.index} info: {json.dumps(scan['info'], indent=2)}")
                    print(f"Available streams: {list(scan['streams'].keys())}")
                else:
                    print(f"No scan found at index {args.index}")
                    
            elif args.command == "list":
                scans = await client.get_all_scans()
                print(f"Total scans: {len(scans)}")
                for i, scan in enumerate(scans):
                    print(f"Scan {i}: {scan.get('info', {}).get('title', 'No title')}")
            
            elif args.command == "clear":
                success = await client.clear_scans()
                if success:
                    print("Successfully cleared all scans from server memory")
                else:
                    print("Failed to clear scans")
                    
        finally:
            await client.disconnect()
    
    asyncio.run(run_command())

