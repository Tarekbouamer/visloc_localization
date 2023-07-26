import functools
import logging
import os
import sys

from loguru import logger
from tabulate import tabulate
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log

# so that calling init_loguru multiple times won't add many handlers


@functools.lru_cache()
def init_logger(log_file=None, *, color=True, name="visloc", abbrev_name=None, file_name=None):
    """
    Initialize logger and set its verbosity level to "DEBUG".
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # formaters
    plain_formatter = logging.Formatter(
        "[%(asctime)s %(name)s]: %(message)s", datefmt="%m/%d %H:%M")
    color_formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
        datefmt="%m/%d %H:%M",
        root_name=name,
        abbrev_name=str(abbrev_name),)

    # console logging
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(color_formatter if color else plain_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # file logging
    if log_file is not None:
        if file_name is not None:
            filename = os.path.join(log_file, file_name + ".txt")
        else:
            filename = os.path.join(log_file, "log.txt")

        file_handler = logging.FileHandler(filename, mode="w")
        file_handler.setFormatter(plain_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


def init_loguru(name="visloc", app_name="VisLoc", log_file=None, file_name=None):

    # log file
    if file_name:
        log_file = os.path.join(log_file, file_name + ".txt")

    logger_format = (
        "<g>{time:YYYY-MM-DD HH:mm}</g>|"
        f"<m>{app_name}</m>|"
        "<level>{level: <8}</level>|"
        "<c>{name}</c>:<c>{function}</c>:<c>{line}</c>|"
        "{extra[ip]} {extra[user]} <level>{message}</level>")

    # ip and user
    logger.configure(extra={"ip": "", "user": ""})  # Default values

    # Remove the default logger configuration
    logger.remove()
    logger.add(log_file, enqueue=True)  # Add a file sink for logging

    # You can add additional sinks for logging, such as console output
    logger.add(sys.stderr, format=logger_format, colorize=True)

    logger.success("init logger")

    return logger


def create_small_table(small_dict, fmt=".2f", coef=1.0):
    """
    """
    keys, values = tuple(zip(*small_dict.items()))

    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=fmt,
        stralign="center",
        numalign="center",
    )
    return table
