# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:58:57 2026

@author: Sebastian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import importlib

import sop_CAPM
importlib.reload(sop_CAPM)


# inputs

position_security = "GOOG"
delta_usd = 15 # en mill

#
benchmark = "^SPX"

#
hedge_securitiess = ["AAPL", "MSFT"]    # esto es de cobertura


# inicializar hedge

hedger = sop_CAPM.hedger(position_security, delta_usd, benchmark, hedge_securitiess)

hedger.compute_betas()

# Según 3_CAPM los betas son:
#  nvda 2,18
#  aapl 1.28
#  msft 1.26

# Según Hedger
# nvda (posicion_beta) 2.18
# aapl (hedge betas) 1.28
# msft (hedge betas) 1.26


# teniendo las betas, queda por invertir las matrices


hedger.compute_optimal_hedge()



