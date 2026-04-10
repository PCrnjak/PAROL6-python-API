"""Async logging using QueueHandler to move I/O off the main thread."""

import logging
import queue
from logging.handlers import QueueHandler, QueueListener


class AsyncLogHandler:
    """Non-blocking logging using QueueHandler + QueueListener.

    Replaces synchronous logging with queue-based async logging.
    Log records are queued immediately (non-blocking) and processed
    by a background thread, preventing logging I/O from introducing
    jitter in timing-critical loops.
    """

    def __init__(self, logger_name: str = "parol6.server.controller"):
        """Initialize the async log handler.

        Args:
            logger_name: Name of the logger to wrap with async handling.
        """
        self._queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
        self._listener: QueueListener | None = None
        self._logger = logging.getLogger(logger_name)
        self._original_handlers: list[logging.Handler] = []
        self._started = False

    def start(self) -> None:
        """Replace logger handlers with queue-based async handler.

        Call once at startup before entering the main loop.
        """
        if self._started:
            return

        # Get the root logger's handlers if this logger has none
        if self._logger.handlers:
            target_handlers = self._logger.handlers[:]
        else:
            # Walk up to find handlers (usually on root logger)
            target_handlers = []
            current: logging.Logger | None = self._logger
            while current is not None:
                if current.handlers:
                    target_handlers = current.handlers[:]
                    break
                if not current.propagate:
                    break
                current = current.parent

        if not target_handlers:
            # No handlers found, nothing to wrap
            return

        # Store original handlers
        self._original_handlers = target_handlers

        # Create QueueHandler
        queue_handler = QueueHandler(self._queue)

        # For the controller logger specifically, replace its handlers
        # and stop propagation so we control all output
        self._logger.handlers = [queue_handler]
        self._logger.propagate = False

        # Start listener thread to process queued records
        self._listener = QueueListener(
            self._queue, *self._original_handlers, respect_handler_level=True
        )
        self._listener.start()
        self._started = True

    def stop(self) -> None:
        """Stop listener and restore original handlers.

        Call at shutdown to ensure all queued messages are flushed.
        """
        if not self._started:
            return

        if self._listener:
            self._listener.stop()
            self._listener = None

        # Restore propagation and clear our handler
        self._logger.handlers = []
        self._logger.propagate = True

        self._original_handlers = []
        self._started = False
