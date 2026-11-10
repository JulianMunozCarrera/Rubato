import os
import sys

# Lets the tests import the bandmate folder without installing anything.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
