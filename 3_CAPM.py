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

import sop_CAPM
importlib.reload(sop_CAPM)


# ----------------------
# Inputs
# ----------------------

# x (independiente)
benchmark = "^SPX"

# y (dependiente)
security = "MSFT"


# ----------------------
# CAPM
# ----------------------

# inicialización de la clase (creación de instancia)
modelo = sop_CAPM.model(benchmark, security)

# Obtención de series de tiempo
modelo.sync_timeseries()

# plot timeseries
#modelo.plot_timesries()

# Calculo de la regresion lineal
modelo.compute_regress()

# PLot regresion
modelo.plot_regress()


