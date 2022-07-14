import logging

"""
Create a singleton logger to act as a global logger.
Usage:

from logger import logger
logger.info('hello world')

And it can be updated from anywhere like this:
import logging
logger.setLevel(logging.DEBUG)
"""

logger = logging.getLogger('softalign')
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s  %(levelname)-5.5s  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
