# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:16:07 2026

@author: Sebastian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

import scipy.optimize as opt

import sop_CAPM
importlib.reload(sop_CAPM)


# --------- Esta 1ra parte es una copia de 4_hedger

# inputs

position_security = "NVDA"
delta_usd = 10 # en mill

#
benchmark = "^SPX"

#
hedge_securitiess = ["GOOG", "AAPL", "MSFT", "SPY" ]    # esto es de cobertura


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

hedge_optimal_weights = hedger.hedge_weights #resultados optimos







# ------------------------- 2da parte


# inputs

betas = hedger.hedge_betas # las betas de cada activo a usar como hedge

target_delta = hedger.posicion_delta_usd  # $$ de la posición

target_beta = hedger.posicion_beta_usd  # $$ en betas




# define la función a minimizar
def cost_function(x, betas, target_delta, target_beta ):
    
    dimensiones = len(x) # q' de dimensiones
    
    # vector de 1s
    deltas = np.ones([dimensiones])
    
    # Crea la función delta, (parte del sistema de ecuación a optimizar) 
    f_delta = (np.transpose(deltas).dot(x).item() + target_delta)**2
    
    # crea la función beta, parte del sistema de ecuación a optimizar
    f_beta = (np.transpose(betas).dot(x) + target_beta)**2
    
    # por ahora sin función de regularización
    #f_penalty = regularization*(np.sum(x**2))    
    f = f_delta + f_beta # f_penalty
    
    return f


# Crea los valores de x, que serán usados para iterar la función

# target delta, es la posición en dolares inicial $$.
# al dividirla por la cantidad de betas, sabemos las "ponderación" aproximada de beta dolares en cada activo de hedge
# 

x0 = - target_delta /len(betas) * np.ones([len(betas), 1])
x0f = x0.flatten()




# computando la optimización
# minimize toma la función
# los posibles valores que tendrá el dominio se indican mediante x0
# y los argumentos o variables a ser usadas en el sistema de ecuación se ingresan mediante args

optimal_result = opt.minimize(fun=cost_function, 
                              x0=x0f,
                              args=(betas, target_delta, target_beta )
                              )

hedge_weights_optimizados = optimal_result.x

print()
print()
print(f"Los pesos de Hedge que minimizan son:")
for i in range(len(hedge_weights_optimizados)):
    print("---")
    print(f"     {hedge_securitiess[i]} {hedge_weights_optimizados[i]}")
    print()







