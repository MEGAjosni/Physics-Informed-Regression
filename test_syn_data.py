# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 00:08:10 2021

@author: jonas
"""

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from models import SIR
from sim_functions import SimulateModel, LeastSquareModel
import numpy as np


x0 = [5600000, 100000, 0]
mp = [0.15, 1/7]

t = np.arange(1, 100)

X = SimulateModel(t, x0, mp)

