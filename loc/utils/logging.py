import functools
import logging
import os
import sys

from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwcfg.pop("root_name") + "."
        self._abbrev_name = kwcfg.pop("abbrev_name", "")
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

# so that calling setup_logger multiple times won't add many handlers


@functools.lru_cache()
def setup_logger(output=None, *, color=True, name="visloc", abbrev_name=None, suffix=None):
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
    if output is not None:
        if suffix is not None:
            filename = os.path.join(output, suffix + ".txt")
        else:
            filename = os.path.join(output, "log.txt")

        file_handler = logging.FileHandler(filename, mode="w")
        file_handler.setFormatter(plain_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger
