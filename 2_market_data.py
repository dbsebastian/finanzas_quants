# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 15:50:45 2026

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

tick_list = ["^SPX", "^SPY", "XLK", "XLV", "XLF", "BTC-USD"]

for ticker in tick_list:
    
    # ^SPX, ^SPY
    
    
    # XLK (sector tecnológico)
    # XLV (salud)
    # XLF (Financiero)
    
    
    path = "h:\\Cursos\\Quants_yt\\Datos"
    path += "\\" + ticker + ".csv"
    
    
    ## funciones
    
    raw_data= pd.read_csv(path)
    t = pd.DataFrame()
    t["date"]= pd.to_datetime(raw_data["Date"], format="mixed", dayfirst=True)
    t["close"]= raw_data["Close"]
    t.sort_values(by="date", ascending=True)
    t["close_previous"]= t["close"].shift(1)
    t["return_close"]= t["close"]/t["close_previous"] - 1
    t= t.dropna()
    t = t.reset_index(drop=True)
    
    
    #### 
    
    
    
    distribucion_tipo = ticker + " | real time" 
    # normal estandart, normal, student, exponencial, chi, uniforme, gamma
    
    
    # aca se generará el input, en una instancia de clase input
    inputs = apoyo_var_market.sim_input()
    
    inputs.tipo_distribucion = distribucion_tipo
    
    inputs.degree_f = 2 # degrees f en student y chi2
    inputs.scale = 5    # degrees f en exponencial
    inputs.mean = 5      # mean en normal
    inputs.std = 10     # std en normal
    
    
    inputs.shape = 5
    
    inputs.size = 10**6
    
    
    
    
    ########################
    # calculos
    ########################
    
    # Crea una instancia, con los inputs deseados
    simulacion_1 = apoyo_var_market.simulador(inputs) 
    
    
    # Genera el vector (con determinada distribución)
    #simulacion_1.generate_vector()  
    simulacion_1.vector = t["return_close"]
    
    inputs.size = len(t.return_close)
    
    
    
    # Computa los stats
    simulacion_1.compute_stats()
    
    # computa el test
    simulacion_1.test_jb()
    
    # computa los plots
    simulacion_1.plot()





