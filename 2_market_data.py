# -*- coding: utf-8 -*-
"""


@author: Sebastian
"""

# imports

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

import apoyo_var_market
importlib.reload(apoyo_var_market)


########################## Tick

ticker = "MA"
t = apoyo_var_market.load_timeseries(ticker)



# Calculos
distribucion = apoyo_var_market.distribution(ticker)

distribucion.load_timeseries()

distribucion.plot_timeseries()

distribucion.compute_stats()

distribucion.plot_histogram()






















