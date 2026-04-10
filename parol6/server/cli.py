"""Command-line interface for the PAROL6 controller."""

import argparse
import logging
import os
import signal

import parol6.config as cfg
from parol6.config import TRACE
from parol6.server.controller import Controller, ControllerConfig

logger = logging.getLogger("parol6.server.cli")


def main() -> int:
    """Main entry point for the controller."""
    # Parse arguments first to get logging level
    parser = argparse.ArgumentParser(description="PAROL6 Robot Controller")
    parser.add_argument("--host", default="0.0.0.0", help="UDP host address")
    parser.add_argument("--port", type=int, default=5001, help="UDP port")
    parser.add_argument("--serial", help="Serial port (e.g., /dev/ttyUSB0 or COM3)")
    parser.add_argument("--baudrate", type=int, default=3000000, help="Serial baudrate")

    # Verbose logging options
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity; -v=INFO, -vv=DEBUG, -vvv=TRACE",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Enable quiet logging (WARNING level)",
    )
    parser.add_argument(
        "--log-level",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set specific log level",
    )
    args = parser.parse_args()

    # Determine log level
    # Precedence:
    #   1) Explicit --log-level
    #   2) Verbose / quiet flags
    #   3) Environment-driven TRACE (PAROL_TRACE=1 via TRACE_ENABLED)
    #   4) Default INFO
    if args.log_level:
        if args.log_level == "TRACE":
            log_level = TRACE
            cfg.TRACE_ENABLED = True
        else:
            log_level = getattr(logging, args.log_level)
    elif args.verbose >= 3:
        log_level = TRACE
        cfg.TRACE_ENABLED = True
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    elif args.quiet:
        log_level = logging.WARNING
    elif cfg.TRACE_ENABLED:
        # Enable TRACE when PAROL_TRACE=1 and no CLI override is given
        log_level = TRACE
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # At INFO and above, push noisy third-party libraries up to WARNING so a
    # clean startup stays quiet. DEBUG/TRACE users still see everything.
    quiet_level = (
        max(log_level, logging.WARNING) if log_level >= logging.INFO else log_level
    )
    for name in ("toppra", "numba", "numba.core", "numba.cuda"):
        logging.getLogger(name).setLevel(quiet_level)

    # Pre-compile numba JIT functions to avoid mid-loop compilation stalls
    from parol6.utils.warmup import warmup_jit

    warmup_jit()

    # Create configuration (env vars may override defaults)
    env_host = os.getenv("PAROL6_CONTROLLER_IP")
    env_port = os.getenv("PAROL6_CONTROLLER_PORT")
    udp_host = env_host.strip() if env_host else args.host
    try:
        udp_port = int(env_port) if env_port else args.port
    except (TypeError, ValueError):
        udp_port = args.port

    logger.debug(f"Controller bind: host={udp_host} port={udp_port}")

    config = ControllerConfig(
        udp_host=udp_host,
        udp_port=udp_port,
        serial_port=args.serial,
        serial_baudrate=args.baudrate,
    )

    # Create and run controller
    controller = None

    def handle_sigterm(signum, frame):
        """Handle SIGTERM signal for graceful shutdown."""
        logger.info("Received SIGTERM, shutting down...")
        if controller:
            controller.stop()
        raise SystemExit(0)

    # Install signal handler for graceful shutdown on SIGTERM
    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        controller = Controller(config)

        if controller.is_initialized():
            try:
                controller.start()
            except KeyboardInterrupt:
                # Controller's "Controller stopped" line is the one summary;
                # no need to announce the interrupt itself.
                pass
            finally:
                controller.stop()
        else:
            logger.error("Controller not properly initialized")
            return 1
    except RuntimeError as e:
        logger.error(f"Failed to create controller: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
