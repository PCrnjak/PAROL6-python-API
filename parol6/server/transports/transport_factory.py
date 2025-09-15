"""
Transport factory for creating appropriate transport instances.

This module provides a factory pattern for creating transport instances
based on configuration and environment. It automatically selects between
real serial, mock serial, or other transport types.
"""

import os
import logging
from typing import Optional, Union

from parol6.server.transports.serial_transport import SerialTransport
from parol6.server.transports.mock_serial_transport import MockSerialTransport

logger = logging.getLogger(__name__)


def is_simulation_mode() -> bool:
    """
    Check if simulation mode is enabled.
    
    Returns:
        True if simulation mode is enabled via environment variable
    """
    fake_serial = str(os.getenv("PAROL6_FAKE_SERIAL", "0")).lower()
    return fake_serial in ("1", "true", "yes", "on")


def create_transport(
    transport_type: Optional[str] = None,
    port: Optional[str] = None,
    baudrate: int = 2000000,
    **kwargs
) -> Union[SerialTransport, MockSerialTransport]:
    """
    Create an appropriate transport instance based on configuration.
    
    The factory will automatically select the appropriate transport:
    - MockSerialTransport if PAROL6_FAKE_SERIAL is set
    - SerialTransport otherwise
    
    Args:
        transport_type: Explicit transport type ('serial', 'mock', or None for auto)
        port: Serial port name (for real serial)
        baudrate: Baud rate for serial communication
        **kwargs: Additional transport-specific parameters
        
    Returns:
        Transport instance (SerialTransport or MockSerialTransport)
    """
    # Determine transport type
    if transport_type is None:
        # Auto-detect based on environment
        if is_simulation_mode():
            transport_type = 'mock'
        else:
            transport_type = 'serial'
    
    # Create appropriate transport
    if transport_type == 'mock':
        logger.info("Creating MockSerialTransport for simulation")
        from parol6.server.transports.mock_serial_transport import MockSerialTransport
        transport = MockSerialTransport(port=port, baudrate=baudrate, **kwargs)
        
    elif transport_type == 'serial':
        logger.info(f"Creating SerialTransport for port: {port}")
        from parol6.server.transports.serial_transport import SerialTransport
        transport = SerialTransport(port=port, baudrate=baudrate, **kwargs)
        
    else:
        raise ValueError(f"Unknown transport type: {transport_type}")
    
    return transport


def create_and_connect_transport(
    transport_type: Optional[str] = None,
    port: Optional[str] = None,
    baudrate: int = 2000000,
    auto_find_port: bool = True,
    **kwargs
) -> Optional[Union[SerialTransport, MockSerialTransport]]:
    """
    Create and connect a transport instance.
    
    This is a convenience function that creates a transport and
    attempts to connect it. For real serial, it can also attempt
    to find and load a saved port.
    
    Args:
        transport_type: Transport type or None for auto-detect
        port: Serial port or None
        baudrate: Baud rate
        auto_find_port: Whether to try loading saved port for serial
        **kwargs: Additional parameters
        
    Returns:
        Connected transport instance or None if connection failed
    """
    # For mock transport, port finding is not needed
    if is_simulation_mode() or transport_type == 'mock':
        transport = create_transport('mock', port=port, baudrate=baudrate, **kwargs)
        if transport.connect():
            return transport
        return None
    
    # For real serial, handle port finding
    if not port and auto_find_port:
        # Try to load saved port
        from parol6.config import get_com_port_with_fallback
        port = get_com_port_with_fallback()
        if port:
            logger.info(f"Using saved serial port: {port}")
    
    # Create transport
    transport = create_transport(transport_type, port=port, baudrate=baudrate, **kwargs)
    
    # Attempt connection if port is known
    if port:
        if transport.connect(port):
            return transport
        else:
            logger.warning(f"Failed to connect to port: {port}")
    
    return transport
