import logging

class CustomFormatter(logging.Formatter):


    grey        = "\x1b[38;21m"
    green       = "\x1b[1;32m"
    yellow      = "\x1b[33;21m"
    red         = "\x1b[31;21m"
    bold_red    = "\x1b[31;1m"
    blue        = "\x1b[1;34m"
    light_blue  = "\x1b[1;36m"
    purple      = "\x1b[1;35m"
    reset       = "\x1b[0m"
    
    format    = "%(levelname)s |  %(asctime)s  %(message)s"
    
    FORMATS = {
        
        logging.DEBUG:    blue      + format + reset,
        logging.INFO:     format,
        logging.WARNING:  yellow    + format + reset,
        logging.ERROR:    red       + format + reset,
        logging.CRITICAL: bold_red  + format + reset
    }

    def format(self, record):
        log_fmt     = self.FORMATS.get(record.levelno)
        formatter   = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        
        return formatter.format(record)