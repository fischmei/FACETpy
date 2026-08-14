"""
Centralized logging configuration for FACETpy.

This module configures Loguru to:
    * log to stderr so users still see live output, and
    * optionally log to a timestamped file when ``FACET_LOG_FILE=1`` is set.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

from loguru import logger

from .config import get_config
from .console import ConsoleMode, configure_console

_LOGGING_CONFIGURED = False
DEFAULT_FACET_CONSOLE_MODE = "classic"


def _resolve_console_mode() -> ConsoleMode:
    value = str(get_config("console_mode") or DEFAULT_FACET_CONSOLE_MODE).strip().lower()
    if value in {"classic", "legacy", "loguru"}:
        return ConsoleMode.CLASSIC
    return ConsoleMode.MODERN


class _InterceptHandler(logging.Handler):
    """
    Handler that intercepts standard library logging and forwards to loguru.

    This allows external libraries (like MNE-Python) that use the standard
    logging module to have their output formatted and routed through loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a logging record to loguru."""
        # Get corresponding loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from who invoked the logging call
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _should_auto_configure() -> bool:
    """Return True unless the user explicitly disables auto logging."""
    return bool(get_config("auto_logging"))


def _file_logging_enabled() -> bool:
    """Return True only when the user explicitly opts in to per-run log files."""
    return bool(get_config("log_file"))


@contextlib.contextmanager
def suppress_stdout() -> Generator[None, None, None]:
    """Context manager to suppress stdout/stderr output (useful for verbose MNE operations)."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _resolve_log_directory() -> Path:
    """Return the directory where log files should be written."""
    config_dir = get_config("log_dir")
    log_dir = Path(config_dir).expanduser() if config_dir else Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@contextlib.contextmanager
def recording_log(path: str | Path) -> Generator[Path, None, None]:
    """Capture one CLI recording's Loguru output in a dedicated file.

    The sink uses the active console level so it contains the same log records
    a classic terminal run receives.  It is synchronous and removed at the end
    of the recording, which prevents recursive or batch runs from mixing EEGs.
    """
    log_path = Path(path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(
        log_path,
        level=str(get_config("log_level")).upper(),
        enqueue=False,
        backtrace=True,
        diagnose=False,
        encoding="utf-8",
        mode="w",
    )
    try:
        yield log_path
    finally:
        logger.remove(sink_id)


def configure_logging(force: bool = False) -> Path | None:
    """
    Configure global logging if it has not been configured yet.

    Parameters
    ----------
    force:
        If True, reconfigure even if logging is already set up.

    Returns
    -------
    Path | None
        The path to the log file sink if logging is configured, otherwise None.
    """
    global _LOGGING_CONFIGURED

    if not force and _LOGGING_CONFIGURED:
        return None

    if not _should_auto_configure():
        return None

    console_level = str(get_config("log_level")).upper()
    console_mode = _resolve_console_mode()
    console_renderer = configure_console(console_mode)

    # Remove any existing sinks so we can guarantee consistent behavior.
    logger.remove()

    # Console sink keeps interactive feedback.
    # enqueue=False keeps logging synchronous so message ordering is preserved —
    # with enqueue=True the background dispatch thread interleaves logger.info()
    # calls with direct print() writes, producing scrambled log output.
    if console_renderer.enabled and console_renderer.requires_sink:
        logger.add(
            console_renderer.log_sink,
            level=console_level,
            enqueue=False,
            backtrace=False,
            diagnose=False,
        )
    else:
        logger.add(sys.stderr, level=console_level, enqueue=False, backtrace=True, diagnose=False)

    # File sink is opt-in: set FACET_LOG_FILE=1 to enable per-run log files.
    log_file = None
    if _file_logging_enabled():
        log_dir = _resolve_log_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = log_dir / f"facet_{timestamp}.log"
        file_level = str(get_config("log_file_level")).upper()
        logger.add(
            log_file,
            level=file_level,
            enqueue=True,
            backtrace=True,
            diagnose=False,
            encoding="utf-8",
        )

    if console_mode is ConsoleMode.MODERN and not console_renderer.enabled:
        logger.warning(
            "Modern console requested but unavailable (missing rich or non-interactive terminal). "
            "Falling back to classic loguru console output."
        )

    # Configure standard library logging to route through loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)

    # Suppress verbose external libraries
    logging.getLogger("mne").setLevel(logging.WARNING)
    logging.getLogger("mne_bids").setLevel(logging.WARNING)
    logging.getLogger("scipy").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)

    # Suppress MNE's verbose print output
    import mne

    mne.set_log_level("WARNING")

    # Suppress NumPy runtime warnings (divide by zero, overflow, invalid values)
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    active_console_mode = console_mode if console_renderer.enabled else ConsoleMode.CLASSIC

    logger.debug(
        "FACETpy logging initialized | console_mode={} | log_file={}",
        active_console_mode.value,
        log_file or "disabled (set FACET_LOG_FILE=1 to enable)",
    )

    _LOGGING_CONFIGURED = True
    return log_file


__all__ = ["configure_logging", "recording_log", "suppress_stdout"]
