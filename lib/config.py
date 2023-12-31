import os
import sys
from easydict import EasyDict

# path
CONF = EasyDict()
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/home/dbi-data5/miyanishi/Project/CityRefer"  # TODO: change this
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
CONF.PATH.SCAN = os.path.join(CONF.PATH.DATA, "sensaturban")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# scannet data
CONF.PATH.SCAN_SCANS = os.path.join(CONF.PATH.SCAN, "scans")
CONF.PATH.SCAN_META = os.path.join(CONF.PATH.SCAN, "meta_data")
CONF.PATH.SCAN_DATA = os.path.join(CONF.PATH.SCAN, "pointgroup_data/balance_split/random-50_crop-250")  # TODO: change this
