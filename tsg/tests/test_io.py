from __future__ import absolute_import
import tsg
import os

DIRNAME = os.path.dirname(__file__)
LEGOS_FILE = os.path.join(DIRNAME, './legos_sample.nc')

def test_open_tsg_from_legos():
    tsg.open_tsg_from_legos(LEGOS_FILE)