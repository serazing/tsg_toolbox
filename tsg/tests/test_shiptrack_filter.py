import tsg
import os

DIRNAME = os.path.dirname(__file__)
LEGOS_FILE = os.path.join(DIRNAME, './legos_sample.nc')


def test_shiptrack_filter():
    data = tsg.open_tsg_from_legos(LEGOS_FILE)
    tsg.shiptrack_filter(data['SST'], cutoff=10e3, win_dt=3, max_break=24)