# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:07:06 2023

@author: anuki
"""

import numpy as np
import time
from hazard_detect import luminance_flash_count

prev = np.random.rand(1200, 1600, 3)
cur = np.random.rand(1200, 1600, 3)

s = time.time()
luminance_flash_count(prev, cur)

print(time.time() - s)

