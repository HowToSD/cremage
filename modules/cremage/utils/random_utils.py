"""
Utility functions for random numbers
"""
import os
import sys
import logging
import random

TMP_DIR = os.path.join(os.path.expanduser("~"), ".cremage", "tmp")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

random_state = None
def safe_random_int():
    """
    Generates consistent random numbers in an environment where random state gets destroyed
    during a program execution.

    Returns:
       int: A randome number
    """
    global random_state
    if random_state is None:
        random_state = random.getstate()

    random.setstate(random_state)
    retval = random.getrandbits(32)
    random_state = random.getstate()
    return retval
