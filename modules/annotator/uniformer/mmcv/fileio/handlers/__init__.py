# Copyright (c) OpenMMLab. All rights reserved.

# Cremage change start
import os
import sys

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
# Cremage change end

from .base import BaseFileHandler
from .json_handler import JsonHandler
from .pickle_handler import PickleHandler
from .yaml_handler import YamlHandler

__all__ = ['BaseFileHandler', 'JsonHandler', 'PickleHandler', 'YamlHandler']
