__version__ = '0.0'


try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Could not import loguru")

try:
    import core
    logger.info(f"Visloc core version {core.__version__}")
except ImportError:
    logger.warning("Could not import visloc_core")

try:
    import matching
    logger.info(f"Visloc matching version {matching.__version__}")
except ImportError:
    logger.warning("Could not import visloc_matching")


import pycolmap

logger.info(f"pycolmap version {pycolmap.__version__}")
