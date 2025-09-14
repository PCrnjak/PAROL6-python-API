"""
UDP transport implementation for PAROL6 controller.

This module handles UDP socket communication, message parsing, and
response handling for the controller's network interface.
"""

from __future__ import annotations

import socket
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class UDPMessage:
    """
    Represents a UDP message received from a client.
    
    Attributes:
        data: The message content as a string
        address: Tuple of (ip, port) of the sender
        timestamp: Unix timestamp when the message was received
    """
    data: str
    address: Tuple[str, int]
    timestamp: float


class UDPTransport:
    """
    Manages UDP socket communication for the controller.
    
    This class handles:
    - Socket creation and binding
    - Non-blocking message reception
    - Response sending
    - Connection management
    """
    
    def __init__(self, ip: str = "127.0.0.1", port: int = 5001, buffer_size: int = 1024):
        """
        Initialize the UDP transport.
        
        Args:
            ip: IP address to bind to
            port: Port number to listen on
            buffer_size: Size of the receive buffer in bytes
        """
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.socket: Optional[socket.socket] = None
        self.message_queue: deque[UDPMessage] = deque(maxlen=100)
        self._running = False
        
    def create_socket(self) -> bool:
        """
        Create and bind the UDP socket.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Set socket to non-blocking mode
            self.socket.setblocking(False)
            
            # Allow address reuse
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to address
            self.socket.bind((self.ip, self.port))
            
            self._running = True
            logger.info(f"UDP socket created and bound to {self.ip}:{self.port}")
            return True
            
        except socket.error as e:
            logger.error(f"Failed to create UDP socket: {e}")
            self.socket = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating UDP socket: {e}")
            self.socket = None
            return False
    
    def close_socket(self) -> None:
        """Close the UDP socket and clean up resources."""
        self._running = False
        
        if self.socket:
            try:
                self.socket.close()
                logger.info("UDP socket closed")
            except Exception as e:
                logger.error(f"Error closing UDP socket: {e}")
            finally:
                self.socket = None
        
        # Clear message queue
        self.message_queue.clear()
    
    def receive_messages(self) -> List[UDPMessage]:
        """
        Receive all available messages from the socket (non-blocking).
        
        Returns:
            List of received UDPMessage objects (may be empty)
        """
        messages = []
        
        if not self.socket or not self._running:
            return messages
        
        # Try to receive all available messages
        while True:
            try:
                # Non-blocking receive
                data, address = self.socket.recvfrom(self.buffer_size)
                
                # Decode the message
                try:
                    message_str = data.decode('utf-8').strip()
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode message from {address}")
                    continue
                
                # Create message object
                msg = UDPMessage(
                    data=message_str,
                    address=address,
                    timestamp=time.time()
                )
                
                messages.append(msg)
                
                # Also add to internal queue for history
                self.message_queue.append(msg)
                
            except socket.error as e:
                # No more data available (EWOULDBLOCK/EAGAIN)
                if e.errno in (11, 35):  # EWOULDBLOCK/EAGAIN
                    break
                else:
                    logger.error(f"Socket error receiving UDP message: {e}")
                    break
            except Exception as e:
                logger.error(f"Unexpected error receiving UDP message: {e}")
                break
        
        return messages
    
    def send_response(self, message: str, address: Tuple[str, int]) -> bool:
        """
        Send a response message to a specific address.
        
        Args:
            message: The message string to send
            address: Tuple of (ip, port) to send to
            
        Returns:
            True if successful, False otherwise
        """
        if not self.socket or not self._running:
            logger.warning("Cannot send response - socket not available")
            return False
        
        try:
            # Encode and send the message
            data = message.encode('utf-8')
            self.socket.sendto(data, address)
            return True
            
        except socket.error as e:
            logger.error(f"Socket error sending UDP response: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending UDP response: {e}")
            return False
    
    def broadcast_message(self, message: str, addresses: List[Tuple[str, int]]) -> int:
        """
        Broadcast a message to multiple addresses.
        
        Args:
            message: The message to broadcast
            addresses: List of (ip, port) tuples to send to
            
        Returns:
            Number of successful sends
        """
        success_count = 0
        
        for address in addresses:
            if self.send_response(message, address):
                success_count += 1
        
        return success_count
    
    def is_running(self) -> bool:
        """
        Check if the transport is running.
        
        Returns:
            True if running, False otherwise
        """
        return self._running and self.socket is not None
    
    def get_recent_messages(self, count: int = 10) -> List[UDPMessage]:
        """
        Get the most recent messages from the internal queue.
        
        Args:
            count: Number of messages to retrieve
            
        Returns:
            List of recent messages (newest first)
        """
        return list(reversed(list(self.message_queue)[-count:]))
    
    def clear_message_queue(self) -> None:
        """Clear the internal message queue."""
        self.message_queue.clear()
    
    def get_socket_info(self) -> dict:
        """
        Get information about the current socket.
        
        Returns:
            Dictionary with socket information
        """
        info = {
            'ip': self.ip,
            'port': self.port,
            'buffer_size': self.buffer_size,
            'running': self._running,
            'socket_open': self.socket is not None,
            'queue_size': len(self.message_queue)
        }
        
        if self.socket:
            try:
                sockname = self.socket.getsockname()
                info['bound_address'] = sockname
            except:
                pass
        
        return info


def create_udp_transport(ip: str = "127.0.0.1", port: int = 5001) -> Optional[UDPTransport]:
    """
    Factory function to create and initialize a UDP transport.
    
    Args:
        ip: IP address to bind to
        port: Port number to listen on
        
    Returns:
        Initialized UDPTransport instance, or None if initialization failed
    """
    transport = UDPTransport(ip, port)
    
    if transport.create_socket():
        return transport
    else:
        return None
