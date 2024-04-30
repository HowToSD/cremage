import os
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
