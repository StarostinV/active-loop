"""Client for sending measurement requests to the push server"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional, Union, Tuple

from torch import Tensor
from active_loop.ports import PUSH_SERVER_PORT
from active_loop.network_utils import read_json_message
from active_loop.logging_utils import setup_logging


class PushClient:
    """Client for interacting with the push server"""
    
    def __init__(self, 
                 host: str = '127.0.0.1', 
                 port: int = PUSH_SERVER_PORT,
                 log_level: str = 'INFO',
                 log_file: Optional[str] = None):
        self.host = host
        self.port = port
        
        # Setup logging
        if log_file is None:
            log_file = f"push_client_{int(time.time())}.log"
            
        self.logger = setup_logging(__name__, log_level, log_file)
        
        self._reader = None
        self._writer = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to the push server"""
        try:
            self.logger.info(f"Connecting to push server at {self.host}:{self.port}...")
            self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
            self._connected = True
            self.logger.info(f"Connected to push server.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to push server: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the push server"""
        if self._connected and self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._connected = False
            self.logger.info("Disconnected from push server")
    
    async def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to the server and get the response"""
        if not self._connected:
            raise ConnectionError("Not connected to push server")
        
        try:
            # Send command
            command_str = json.dumps(command)
            self.logger.debug(f"Sending command: {command_str}")
            self._writer.write(command_str.encode())
            await self._writer.drain()
            
            # Get response using shared utility function
            response = await read_json_message(self._reader)
            
            self.logger.debug(f"Received response: {response}")
            return response
        
        except Exception as e:
            self.logger.error(f"Error communicating with server: {e}")
            raise
    
    async def request_measurement(self, 
                                candidates: Tensor,
                                clear_queue: bool = True,
                                reopen_env: bool = True,
                                **kwargs: Any
                                ) -> Dict[str, Any]:
        """Request a measurement at the specified x positions"""
        # Convert single position to list if needed
        assert candidates.ndim == 2

        if candidates.shape[1] == 1:
            x_positions = candidates.flatten().tolist()
            om_positions = None
            self.logger.info(f"Requesting measurement at x positions: {x_positions}")

        elif candidates.shape[1] == 2:
            x_positions, om_positions = candidates.split(1, dim=-1)
            x_positions = x_positions.flatten().tolist()
            om_positions = om_positions.flatten().tolist()
            self.logger.info(f"Requesting measurement at x positions: {x_positions} and om positions: {om_positions}")
        else:
            raise ValueError(f"Invalid number of dimensions: {candidates.shape[1]}")
                            
        command = {
            'command': 'measure',
            'x_positions': x_positions,
            'om_positions': om_positions,
            'clear_queue': clear_queue,
            'reopen_env': reopen_env,
            **kwargs
        }
        
        response = await self._send_command(command)
        
        if 'error' in response:
            self.logger.error(f"Error requesting measurement: {response['error']}")
        else:
            self.logger.info(f"Measurement request queued with ID: {response.get('request_id')}")
            
        return response
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get the status of the measurement queue"""
        self.logger.info("Getting queue status")
        
        command = {
            'command': 'queue_status'
        }
        
        response = await self._send_command(command)
        self.logger.info(f"Queue size: {response.get('queue_size', 0)}, Total requests: {response.get('total_requests', 0)}")
        
        return response
    
    async def get_request_info(self, request_id: int) -> Dict[str, Any]:
        """Get information about a specific measurement request"""
        self.logger.info(f"Getting info for request ID: {request_id}")
        
        command = {
            'command': 'get_request',
            'idx': request_id
        }
        
        response = await self._send_command(command)
        
        if 'error' in response:
            self.logger.error(f"Error getting request info: {response['error']}")
        else:
            self.logger.info(f"Retrieved info for request ID: {request_id}")
            
        return response
        
    async def clear_measurement_queue(self) -> Dict[str, Any]:
        """Clear all pending measurement requests from the queue"""
        self.logger.info("Clearing measurement queue")
        
        command = {
            'command': 'clear_queue'
        }
        
        response = await self._send_command(command)
        
        if 'error' in response:
            self.logger.error(f"Error clearing measurement queue: {response['error']}")
        else:
            cleared_count = response.get('cleared_count', 0)
            self.logger.info(f"Successfully cleared {cleared_count} pending requests from the queue")
            
        return response
    
    async def reopen_environment(self) -> Dict[str, Any]:
        """Reopen the environment on the server"""
        self.logger.info("Reopening environment")
        
        command = {
            'command': 'reopen_environment'
        }
        
        response = await self._send_command(command)
        
        if 'error' in response:
            self.logger.error(f"Error reopening environment: {response['error']}")
        else:
            self.logger.info("Environment reopened successfully")
            
        return response


# Example usage
async def example_usage():
    client = PushClient()
    if await client.connect():
        try:
            # Request a measurement at multiple positions
            response = await client.request_measurement(
                x_positions=[-10, -5, 0, 5, 10],
                tt_max=1.2,
                num_points=100
            )
            print(f"Measurement response: {response}")
            
            # Check queue status
            status = await client.get_queue_status()
            print(f"Queue status: {status}")
            
            # Clear the measurement queue
            clear_result = await client.clear_measurement_queue()
            print(f"Queue cleared: {clear_result}")
            
        finally:
            await client.disconnect()


if __name__ == "__main__":
    asyncio.run(example_usage())


def main():
    """Entry point for the push client command-line tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Push Client for XRR measurements")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=PUSH_SERVER_PORT, help="Server port")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                      help="Logging level")
    parser.add_argument("--log-file", default=None, help="File to save logs to")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)
    
    # Measure command
    measure_parser = subparsers.add_parser("measure", help="Request a measurement")
    measure_parser.add_argument("--x-pos", type=float, nargs="+", required=True, 
                             help="X positions for measurement (space-separated list)")
    measure_parser.add_argument("--tt-max", type=float, default=1.0, help="Maximum tt angle")
    measure_parser.add_argument("--num-points", type=int, default=64, help="Number of measurement points")
    measure_parser.add_argument("--no-clear-queue", action="store_true", help="Don't clear the queue before measuring")
    measure_parser.add_argument("--no-close-env", action="store_true", help="Don't close the environment after measuring")
    
    # Status command
    subparsers.add_parser("status", help="Get queue status")
    
    # Request info command
    info_parser = subparsers.add_parser("info", help="Get request information")
    info_parser.add_argument("request_id", type=int, help="Request ID to get information for")
    
    # Range command for measuring at range of positions
    range_parser = subparsers.add_parser("range", help="Measure over a range of positions")
    range_parser.add_argument("--start", type=float, required=True, help="Start position")
    range_parser.add_argument("--end", type=float, required=True, help="End position")
    range_parser.add_argument("--step", type=float, required=True, help="Step size")
    range_parser.add_argument("--tt-max", type=float, default=1.0, help="Maximum tt angle")
    range_parser.add_argument("--num-points", type=int, default=64, help="Number of measurement points")
    range_parser.add_argument("--no-clear-queue", action="store_true", help="Don't clear the queue before measuring")
    range_parser.add_argument("--no-close-env", action="store_true", help="Don't close the environment after measuring")
    
    # Clear queue command
    subparsers.add_parser("clear", help="Clear all pending measurement requests from the queue")
    
    args = parser.parse_args()
    
    async def run_command():
        client = PushClient(
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            log_file=args.log_file
        )
        
        if not await client.connect():
            print(f"Failed to connect to server at {args.host}:{args.port}")
            return
        
        try:
            if args.command == "measure":
                response = await client.request_measurement(
                    x_positions=args.x_pos,
                    tt_max=args.tt_max,
                    num_points=args.num_points,
                    clear_queue=not args.no_clear_queue,
                    close_environment=not args.no_close_env
                )
                
                if 'error' in response:
                    print(f"Error: {response['error']}")
                else:
                    print(f"Measurement request queued with ID: {response.get('request_id')}")
                    print(f"Queue size: {response.get('queue_size')}")
                    
            elif args.command == "status":
                status = await client.get_queue_status()
                print(f"Queue size: {status.get('queue_size', 0)}")
                print(f"Total requests: {status.get('total_requests', 0)}")
                
            elif args.command == "info":
                info = await client.get_request_info(args.request_id)
                if 'error' in info:
                    print(f"Error: {info['error']}")
                elif 'request' in info:
                    request = info['request']
                    print(f"Request ID: {request.get('idx')}")
                    print(f"X positions: {request.get('x_positions')}")
                    print(f"TT max: {request.get('tt_max')}")
                    print(f"Num points: {request.get('num_points')}")
                    print(f"Timestamp: {request.get('timestamp')}")
                else:
                    print("No information available")
                    
            elif args.command == "range":
                # Generate position list
                positions = []
                current = args.start
                while current <= args.end:
                    positions.append(current)
                    current += args.step
                
                # Request measurement
                print(f"Requesting measurement at {len(positions)} positions: {positions}")
                response = await client.request_measurement(
                    x_positions=positions,
                    tt_max=args.tt_max,
                    num_points=args.num_points,
                    clear_queue=not args.no_clear_queue,
                    close_environment=not args.no_close_env
                )
                
                if 'error' in response:
                    print(f"Error: {response['error']}")
                else:
                    print(f"Measurement request queued with ID: {response.get('request_id')}")
                    print(f"Queue size: {response.get('queue_size')}")
            
            elif args.command == "clear":
                response = await client.clear_measurement_queue()
                if 'error' in response:
                    print(f"Error: {response['error']}")
                else:
                    print(f"Successfully cleared {response.get('cleared_count', 0)} pending requests from the queue")
                    
        finally:
            await client.disconnect()
    
    asyncio.run(run_command())
