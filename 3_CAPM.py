# -*- coding: utf-8 -*-
"""
@author: Sebastian
"""

# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

import apoyo_var_market
importlib.reload(apoyo_var_market)

# Inputs

# x
benchmark = "XLV"
# es la variable x (independiente)


# y
security = "LLY"
# es la variable y (dependiente)


# CAPM

# inicialización de la clase (creación de instancia)
capm_1 = apoyo_var_market.capm(benchmark, security)

# Obtención de series de tiempo
capm_1.sync_timeseries()

# plot timeseries
capm_1.plot_timesries()


# Calculo de la regresion lineal
capm_1.compute_regress()


# PLot regresion

capm_1.plot_regress()
