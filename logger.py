import logging
logging.basicConfig(format="%(levelname)-s:%(name)-s.%(funcName)-s: %(message)s")

import os

def get_logger(name, level):
    log = logging.getLogger(os.path.basename(name))
    log.setLevel(getattr(logging, level.upper()))
    return log
