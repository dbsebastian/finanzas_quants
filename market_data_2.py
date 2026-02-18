# -*- coding: utf-8 -*-

"""
Iterar sobre archivos


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


tick_list = list()

jb_list = list()
p_valor_list = list()
resultado_list = list()

sharpe_list = list()


for tick in ticks_archivos:
    
    if "txt" in tick:
        pass
    else:    
        tick = tick.split(".")[0]
        
        t = apoyo_var_market.load_timeseries(tick)
    
        # computacion
        dist = apoyo_var_market.distribution(tick)
        
        dist.load_timeseries()
        
        #distribucion.plot_timeseries()
        
        dist.compute_stats()
        
        #distribucion.plot_histogram()
        
        
        
        # computa el test
        jb_test = dist.test_jb
        p_valor = dist.p_valor
        
        # test JB
        tick_list.append(tick)
        
        jb_list.append(jb_test)
        p_valor_list.append(p_valor)
        
        resultado_list.append(dist.es_normal)
        
        # ratio Sharpe
        
        sharpe_list.append(dist.sharpe_ratio)
        
        
df_dict = {"ticks":tick_list, 
           "jb_test":jb_list,
           "p_valor":p_valor_list,
           "resultado":resultado_list,
           "Sharpe_ratio":sharpe_list,
           }

df = pd.DataFrame.from_dict(df_dict)

df= df.sort_values(by="Sharpe_ratio", ascending=False)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
