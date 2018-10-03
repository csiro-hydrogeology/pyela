import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime

pkg_dir = os.path.join(os.path.dirname(__file__),'..')

sys.path.append(pkg_dir)

from ela.textproc import *

def test_litho_marker_detection():
    prim_classes = {
        'sand': 0,
        'clay': 1,
        'sandstone': 2
    }
    adjective_classes = {
        'sandy': 0,
        'clayey': 1,
        'choc': 3
    }
    sentences = [
        ['sand','with','greyish','clay'],
        ['clayey','blueish','sandstone','with','coarse','choc','chips']
    ]
    prims= v_find_primary_lithology(sentences, prim_classes)
    assert prims[0] == 0
    assert prims[1] == 2
    seconds = v_find_secondary_lithology(sentences, prims, adjective_classes, prim_classes)
    assert seconds[0] == 1
    assert seconds[1] == 1 # clayey, not choc

