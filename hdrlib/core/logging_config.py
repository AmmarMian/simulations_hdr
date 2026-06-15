# Centralised logging setup for 2-detection scripts and library.
#
# Usage in library modules:
#   import logging
#   logger = logging.getLogger(__name__)
#   logger.debug("...")
#
# Usage in CLI scripts (call once at startup, typically inside setup_run):
#   from src.logging_config import setup_logging, log_arguments
#   setup_logging(quiet=args.quiet)
#   log_arguments(args)

import logging
from argparse import Namespace

from rich.logging import RichHandler


def setup_logging(quiet: bool = False, debug: bool = False) -> None:
    """Configure the root logger with a RichHandler.

    Call **once** per process at script startup (e.g. inside ``setup_run``).
    All subsequent ``logging.getLogger(name)`` calls in any module will
    inherit this configuration automatically.

    Parameters
    ----------
    quiet : bool
        When True, sets the root level to WARNING so INFO/DEBUG messages
        are suppressed.  When False (default), sets the level to INFO.
    debug : bool
        When True, sets the root level to DEBUG. Overrides quiet=True.
    """
    if debug:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
        ],
        force=True,  # override any handlers added before this call
    )


def log_arguments(args: Namespace) -> None:
    """Log all command-line arguments in a formatted way.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments from argparse.
    """
    logger = logging.getLogger(__name__)
    arg_dict = vars(args)
    max_key_len = max(len(k) for k in arg_dict.keys()) if arg_dict else 0

    logger.info("Configuration:")
    for key, value in arg_dict.items():
        logger.info(f"  {key:<{max_key_len}} = {value}")
