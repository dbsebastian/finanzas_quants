# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:16:07 2026

@author: Sebastian
"""

import importlib
import sop_CAPM
importlib.reload(sop_CAPM)


# --------- Esta 1ra parte es una copia de 4_hedger

# inputs
position_security = "V"

#
benchmark = "^SPX"

#
# hedge_universe = [ "AAPL", "MSFT", "NVDA",  \
#                 "AMZN", "GOOG", "META",     \
#                 "NFLX", "SPY", "XLK", "XLF"]    

hedge_universe = ["BRK-B", "JPM", "V", "MA", "BAC", "MS", "GS", "BLK", "SPY", "XLF"]


# ------ Evaluacion de corr, y elección de mejores Hedges

df = sop_CAPM.dataframe_corr_beta(benchmark, position_security, hedge_universe)



# ------ Evaluacion de coberturas
hedge_secu_by_corr = df[df["correlations"]>0.6]["hedge_security"].values
hedge_securitiess = ["MA", "SPY"]
delta_usd = 10 # en mill
reg_param = 0.001

hedger = sop_CAPM.hedger(position_security, delta_usd, benchmark, hedge_securitiess)
hedger.compute_betas()
hedger.compute_optimal_hedge()



# ------ Optimización con penalizacion



optimal_result = hedger.compute_hedge_weights(reg_param)







