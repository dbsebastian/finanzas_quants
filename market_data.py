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


##########
## path
##########

#os.getcwd()

# tick_list = ["^SPX", "^SPY", "XLK", "XLV", "XLF", "BTC-USD"]


    
# ^SPX, ^SPY


# XLK (sector tecnológico)
# XLV (salud)
# XLF (Financier"o)

ticks_archivos = os.listdir("h:\\Cursos\\Quants_yt\\Datos")
ticks_archivos










########################## funciones

ticker = "SPY"
t = apoyo_var_market.load_timeseries(ticker)



# computacion
distribucion = apoyo_var_market.distribution(ticker)

distribucion.load_timeseries()

distribucion.plot_timeseries()

distribucion.compute_stats()

distribucion.plot_histogram()






