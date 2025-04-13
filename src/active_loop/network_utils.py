"""Utility functions for network operations"""

import asyncio
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

async def read_json_message(reader: asyncio.StreamReader, 
                           max_size: int = 10 * 1024 * 1024,
                           chunk_size: int = 16384) -> Dict[str, Any]:
    """Read a complete JSON message from a StreamReader
    
    Args:
        reader: The StreamReader to read from
        max_size: Maximum allowed message size in bytes
        chunk_size: Size of chunks to read at once
        
    Returns:
        The parsed JSON message as a dictionary
        
    Raises:
        ConnectionError: If the connection was closed
        ValueError: If the message exceeds max_size
        json.JSONDecodeError: For malformed JSON
    """
    buffer = bytearray()
    
    while True:
        chunk = await reader.read(chunk_size)
        if not chunk:
            if not buffer:
                raise ConnectionError("Connection closed by server")
            break
        
        buffer.extend(chunk)
        
        if len(buffer) > max_size:
            raise ValueError(f"Message size exceeds maximum allowed size ({max_size} bytes)")
            
        # Try to parse what we have so far to see if it's complete
        try:
            message = json.loads(buffer.decode())
            # If we get here, the JSON is valid and complete
            return message
        except json.JSONDecodeError as e:
            # If we have an unterminated string, we need more data
            if "Unterminated string" in str(e) or "Expecting" in str(e):
                # Continue reading more data
                continue
            else:
                # Some other JSON error - if we've read the full message but it's invalid
                if not chunk or len(chunk) < chunk_size:
                    raise
                # Otherwise continue reading 