import argparse
import logging

from pathlib import Path
import sys


def get_kwargs() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l",
        "--logger_level",
        metavar="<logger_level>",
        type=int,
        help=("NOTSET: 0, DEBUG: 10, INFO: 20, "
              "WARNING: 30, ERROR: 40, CRITICAL: 50"),
        default=20,
    )
    return parser


def get_logger(
        logger_name: str | None = None,
        level: int = logging.DEBUG) -> logging.Logger:

    formatter = logging.Formatter(
        "%(levelname)-8s [%(asctime)s] %(name)s:%(lineno)d: %(message)s"
    )

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    return logger