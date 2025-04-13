"""Server that pushes measurement commands to the beamline"""

import asyncio
import json
import logging
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.zmq import REManagerAPI

from active_loop.ports import PUSH_SERVER_PORT, PROD_REDIS_CONFIG, TEST_REDIS_CONFIG
from active_loop.network_utils import read_json_message
from active_loop.logging_utils import setup_logging
from active_loop.bplan_commands import (
    CorrectAlignmentX,
    SetX,
    XOMMapScan,
    MeasureFullXRR,
    DEFAULT_TT_MAX,
    DEFAULT_TT_MIN,
    DEFAULT_GPOS1,
    DEFAULT_GPOS2,
    DEFAULT_LPOS1,
    DEFAULT_LPOS2,
    DEFAULT_NUM_POINTS,
    DEFAULT_ACQUISITION_TIME,
)

@dataclass
class MeasurementRequest:
    """Data structure to hold measurement request information"""
    x_positions: List[float]
    om_positions: Optional[List[float]] = None
    tt_max: float = DEFAULT_TT_MAX
    tt_min: float = DEFAULT_TT_MIN
    gpos1: float = DEFAULT_GPOS1
    gpos2: float = DEFAULT_GPOS2
    lpos1: float = DEFAULT_LPOS1
    lpos2: float = DEFAULT_LPOS2
    num_points: int = DEFAULT_NUM_POINTS
    clear_queue: bool = True
    reopen_env: bool = True
    timestamp: float = None
    acquisition_time: float = DEFAULT_ACQUISITION_TIME


    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class PushServer:
    """Server that handles measurement requests"""

    def __init__(self, 
                 host: str = '127.0.0.1', 
                 port: int = PUSH_SERVER_PORT, 
                 use_prod: bool = True, 
                 log_level: str = 'INFO',
                 log_file: Optional[str] = None,
                 current_x: float = 0.0,
                 test_mode: bool = False,
                 ):
        self.host = host
        self.port = port
        self.use_prod = use_prod
        self.current_x = current_x
        self.test_mode = test_mode
        
        # Redis configuration
        self.redis_config = PROD_REDIS_CONFIG if use_prod else TEST_REDIS_CONFIG
        
        # Task management
        self.measurement_queue = asyncio.Queue()
        self.queue_lock = asyncio.Lock()  # Lock for queue operations
        self.processing_task = None
        self.running = False
        
        # Request history for monitoring
        self.requests = []
        
        # For log file default name
        if log_file is None:
            log_file = f"push_server_{int(time.time())}.log"
            
        # Set up logging
        self.logger = setup_logging(__name__, log_level, log_file)

        self.api = None

        if not self.test_mode:
            self.connect_api()
        else:
            self.logger.info("Running in test mode - API connection skipped")
    
    def connect_api(self):
        self.api = REManagerAPI(
            zmq_control_addr=self.redis_config.control_url,
            zmq_info_addr=self.redis_config.info_url
        )
        self.reopen_environment()
        self.logger.info("Connected to API")
        
    async def start(self):
        """Start the server and processing loop"""
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        self.running = True
        self.processing_task = asyncio.create_task(self.process_measurements())
        self.logger.info(f"Push server started on {self.host}:{self.port}")
        async with self.server:
            await self.server.serve_forever()

    async def stop(self):
        """Stop the server and processing loop"""
        self.running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        self.server.close()
        await self.server.wait_closed()
        self.logger.info("Push server stopped")

    async def process_measurements(self):
        """Process measurement requests from the queue"""
        self.logger.info("Starting measurement processing loop")
        
        while self.running:
            try:
                # Acquire lock before getting from queue
                async with self.queue_lock:
                    # Use get_nowait instead of get to avoid blocking while holding the lock
                    try:
                        request = self.measurement_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        # If queue is empty, release lock and wait
                        request = None
                
                if request is None:
                    # Wait a bit before checking again
                    await asyncio.sleep(0.1)
                    continue
                
                self.logger.info(f"Processing measurement request with {len(request.x_positions)} positions")
                
                try:
                    # Process the measurement
                    await self.execute_measurement(request)
                    self.logger.info(f"Completed measurement request")
                except Exception as e:
                    self.logger.error(f"Error processing measurement: {e}")
                
                # Mark task as done
                self.measurement_queue.task_done()
            except asyncio.CancelledError:
                self.logger.info("Measurement processing loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in measurement loop: {e}")
                await asyncio.sleep(1)  # Avoid tight error loops


    def request2bplans(self, request: MeasurementRequest) -> List[BPlan]:
        """Convert a measurement request to a list of BPlans"""
        # Create measurement commands
        if request.om_positions is None:
            return self.process_request_with_full_xrr(request)
        else:
            return self.process_request_with_xom_map(request)
        
    def process_request_with_xom_map(self, request: MeasurementRequest) -> List[BPlan]:
        x = request.x_positions
        om = request.om_positions
        integration_time = request.acquisition_time
        gpos1 = request.gpos1
        gpos2 = request.gpos2
        lpos1 = request.lpos1
        lpos2 = request.lpos2

        # sort points so that om is increasing
        if len(om) > 1:
            sorted_idx = np.argsort(np.array(om))
            x = np.array(x)[sorted_idx].tolist()
            om = np.array(om)[sorted_idx].tolist()

        fscan = XOMMapScan(
            x=x,
            om=om,
            integration_time=integration_time,
            gpos1=gpos1,
            gpos2=gpos2,
            lpos1=lpos1,
            lpos2=lpos2
        )
        return [fscan()]

    def process_request_with_full_xrr(self, request: MeasurementRequest) -> List[BPlan]:
        correct_alignment = CorrectAlignmentX(
            gpos1=request.gpos1,
            gpos2=request.gpos2,
            lpos1=request.lpos1,
            lpos2=request.lpos2
        )
        set_x = SetX(correct_alignment)

        bplans = []
    
        # Add all measurement points to the queue
        for x in request.x_positions:
            self.logger.info(f"Adding measurements for x = {x}")
            if x != self.current_x:
                self.logger.info(f"Setting x to {x}")
                self.current_x = x
                bplans = list(set_x(x))
            else:
                self.logger.info(f"X is already at {x}, skipping set_x")

            bplans += list(MeasureFullXRR(
                tt_max=request.tt_max,
                num_points=request.num_points,
                tt_min=request.tt_min,
                acquisition_time=request.acquisition_time,
            ))

        return bplans
    

    async def execute_measurement(self, request: MeasurementRequest):
        """Execute a measurement request"""
        self.logger.info(f"Executing measurement at positions: {request.x_positions}")
        
        # Get BPlans regardless of test mode
        bplans = self.request2bplans(request)
        
        if self.test_mode:
            # In test mode, just print the BPlans instead of submitting to API
            self.logger.info("TEST MODE: Would execute the following BPlans:")
            for idx, bplan in enumerate(bplans):
                self.logger.info(f"BPlan {idx+1}: {bplan}")
            return
            
        # Create API client
        api = self.api
        
        try:
            # Handle environment and queue setup
            if request.clear_queue:
                self.logger.info("Clearing queue")
                api.queue_clear()
            
            if request.reopen_env:
                try:
                    self.reopen_environment()
                except Exception as e:
                    self.logger.warning(f"Error reopening environment: {e}")

            item_uids = []

            for bplan in bplans:
                try:
                    result = api.item_add(bplan)
                    item_uid = result['item']['item_uid']
                    item_uids.append(item_uid)
                    self.logger.info(f"Added plan: {bplan.name} with UID: {item_uid}")
                except Exception as e:
                    self.logger.error(f"Failed to add measurement plan: {e}")
            # Start the queue
            if item_uids:
                self.logger.info("Starting queue execution")
                api.queue_start()
                
                # Wait for queue to complete
                while True:
                    status = api.status()
                    if status.get('manager_state') == 'idle':
                        break
                    await asyncio.sleep(5)  # Check every 5 seconds
                
                self.logger.info("Queue execution completed")
            else:
                self.logger.warning("No measurement plans were added to the queue")
                
        except Exception as e:
            self.logger.error(f"Error during measurement execution: {e}")
            raise

    async def clear_measurement_queue(self) -> int:
        """Clear all pending measurement requests from the queue
        
        Returns:
            Number of cleared items
        """
        # Acquire lock to ensure exclusive access to the queue
        async with self.queue_lock:
            # Get current queue size
            size = self.measurement_queue.qsize()
            self.logger.info(f"Clearing measurement queue with {size} pending requests")
            
            # Empty the queue by getting all items without processing them
            while not self.measurement_queue.empty():
                try:
                    self.measurement_queue.get_nowait()
                    self.measurement_queue.task_done()
                except asyncio.QueueEmpty:
                    # This shouldn't happen since we check empty() first, but just in case
                    break
            
            # Clear the API queue too
            try:
                self.api.queue_clear()
                self.logger.info("API queue cleared")
            except Exception as e:
                self.logger.error(f"Error clearing API queue: {e}")
            
            self.logger.info(f"Measurement queue cleared, removed {size} pending requests")
            return size

    def reopen_environment(self):
        """Reopen the environment on the server"""
        if self.test_mode:
            self.logger.info("TEST MODE: Would reopen environment")
            return
            
        self.logger.info("Reopening environment")
        try:
            self.api.environment_close()
            self.api.wait_for_idle()
        except Exception as e:
            self.logger.warning(f"Error closing environment: {e}")

        try:
            self.api.environment_open()
            self.api.wait_for_idle()
        except Exception as e:
            self.logger.warning(f"Error opening environment: {e}")
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a client connection"""
        addr = writer.get_extra_info('peername')
        self.logger.info(f"Client connected: {addr}")
        
        try:
            while True:
                try:
                    # Use the utility function to read a complete JSON message
                    request = await read_json_message(reader)
                    
                    command = request.get('command')
                    response = {}
                    
                    if command == 'measure':
                        # Create measurement request
                        x_positions = request.get('x_positions', [])
                        if not x_positions:
                            response = {'error': 'No x positions provided'}
                        else:
                            measurement = MeasurementRequest(
                                x_positions=x_positions,
                                om_positions=request.get('om_positions', None),
                                tt_max=request.get('tt_max', DEFAULT_TT_MAX),
                                tt_min=request.get('tt_min', DEFAULT_TT_MIN),
                                gpos1=request.get('gpos1', DEFAULT_GPOS1),
                                gpos2=request.get('gpos2', DEFAULT_GPOS2),
                                lpos1=request.get('lpos1', DEFAULT_LPOS1),
                                lpos2=request.get('lpos2', DEFAULT_LPOS2),
                                num_points=request.get('num_points', DEFAULT_NUM_POINTS),
                                acquisition_time=request.get('acquisition_time', DEFAULT_ACQUISITION_TIME),
                                clear_queue=request.get('clear_queue', True),
                                reopen_env=request.get('reopen_env', True)
                            )
                            
                            # Add to queue with lock protection
                            self.requests.append(measurement)
                            async with self.queue_lock:
                                await self.measurement_queue.put(measurement)
                                queue_size = self.measurement_queue.qsize()
                            
                            response = {
                                'status': 'queued',
                                'queue_size': queue_size,
                                'request_id': len(self.requests) - 1
                            }
                    
                    elif command == 'queue_status':
                        async with self.queue_lock:
                            queue_size = self.measurement_queue.qsize()
                        
                        response = {
                            'queue_size': queue_size,
                            'total_requests': len(self.requests)
                        }
                    
                    elif command == 'get_request':
                        idx = request.get('idx', -1)
                        if 0 <= idx < len(self.requests):
                            req = self.requests[idx]
                            response = {
                                'request': {
                                    'x_positions': req.x_positions,
                                    'tt_max': req.tt_max,
                                    'num_points': req.num_points,
                                    'timestamp': req.timestamp,
                                    'idx': idx
                                }
                            }
                        else:
                            response = {'error': f'Invalid request index: {idx}'}
                    
                    elif command == 'clear_queue':
                        cleared = await self.clear_measurement_queue()
                        response = {
                            'success': True,
                            'cleared_count': cleared,
                            'message': f'Cleared {cleared} pending requests'
                        }
                    
                    elif command == 'reopen_environment':
                        try:
                            self.reopen_environment()
                            response = {
                                'success': True,
                                'message': 'Environment reopened'
                            }
                        except Exception as e:
                            response = {
                                'error': f'Error reopening environment: {str(e)}'
                            }
                    
                    else:
                        response = {'error': f'Unknown command: {command}'}
                    
                    # Send response
                    writer.write(json.dumps(response).encode())
                    await writer.drain()
                    
                except json.JSONDecodeError:
                    writer.write(json.dumps({'error': 'Invalid JSON'}).encode())
                    await writer.drain()
                except Exception as e:
                    self.logger.error(f"Error processing request: {e}")
                    writer.write(json.dumps({'error': str(e)}).encode())
                    await writer.drain()
        
        except Exception as e:
            self.logger.error(f"Client connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            self.logger.info(f"Client disconnected: {addr}")


if __name__ == "__main__":
    server = PushServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        pass


def main():
    """Entry point for the push server command-line tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Push Server for XRR measurements")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=PUSH_SERVER_PORT, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                      help="Logging level")
    parser.add_argument("--log-file", default=None, help="File to save logs to")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no API connection)")
    args = parser.parse_args()
    
    server = PushServer(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        log_file=args.log_file,
        test_mode=args.test
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nShutting down server...")
