# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:48:39 2026

@author: Sebastian
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import importlib

import sop_market_data
importlib.reload(sop_market_data)







class model:
    
    # constructor
    def __init__(self, benchmark, security, decimals= 5):
        self.benchmark = benchmark
        self.security = security
        self.decimals = decimals
        #
        self.timeseries = None
        #
        self.x = None
        self.y = None
        self.alpha = None
        self.beta = None
        self.p_value = None
        self.null_hyp = None
        self.r_square = None
        self.predict_y = None
        #
        
    
    def sync_timeseries(self):
        self.timeseries = sop_market_data.sync_tickers_timeseries(self.benchmark, self.security)
    
    
    def plot_timesries(self):
        plt.figure(figsize=(12,5))
        plt.title(f'Timeserie de {self.security} y {self.benchmark}')
        plt.ylabel("Precios")
        plt.xlabel("Date")
        #ejes
        ax = plt.gca()
        #sub ejes
        ax1 = self.timeseries.plot(kind="line", x="date", y="close_x", ax=ax,
                                   label=f"{self.benchmark}", color="blue",
                                   alpha=0.8, grid=True,
                                   )

        ax2 = self.timeseries.plot(kind="line", x='date', y='close_y',
                                   secondary_y=True, # eje secundario
                                   ax=ax, label=f"{self.security}", color='red',
                                   alpha =0.8,
                                   )
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()
        
    
    def compute_regress(self):
        
        self.x = self.timeseries["return_x"].values
        self.y = self.timeseries["return_y"].values
        
        beta, intercept, r2, pval, stderr = stats.linregress(self.x, self.y)
        
        self.alpha = np.round(intercept, self.decimals)
        self.beta = np.round(beta, self.decimals)
        self.p_value = np.round(pval, self.decimals)
        self.null_hyp = pval > 0.05
        self.r_square = np.round(r2, self.decimals)
        
        # regresion lineal graph
        self.predict_y = self.alpha + self.beta * self.x


    def plot_regress(self):        
        
        
        plt.scatter(x=self.x, y=self.y, color="r", alpha=0.3) 
        plt.plot(self.x, self.predict_y, color="b") 
        
        decimals = 4
        
        str_title = f"Scatterplot de Returns"
        
        str_title += f"\n Regresión lineal, dónde: Security: {self.security} y Benchmark: {self.benchmark} "
        
        str_title += f"\n Alpha: {np.round(self.alpha, decimals)} |  Beta: {np.round(self.beta, decimals)}"
        
        str_title += f"\n un P-value: {np.round(self.p_value, decimals)}, null hypotesis: {self.null_hyp}"
        
        str_title += f"\n y una Correlación: {np.round(self.r_square, decimals)}  y  R-square: {np.round(self.r_square**2, decimals)}"
        
        plt.title(str_title)
        
        plt.xlabel(self.benchmark)
        plt.ylabel(self.security)



def compute_beta(benchmark, security):
    m = model(benchmark, security)
    m.sync_timeseries()
    m.compute_regress()
    
    return m.beta



class hedger:
    
    def __init__(self, position_security, posicion_delta_usd, benchmark, hedge_tics):
        
        self.position_security = position_security
        self.posicion_delta_usd = posicion_delta_usd
        self.posicion_beta = None
        self.posicion_beta_usd = None
        self.benchmark = benchmark
        self.hedge_securitiess = hedge_tics
        self.hedge_betas = list()
        self.hedge_weights = None
        self.hedge_delta_usd = None
        self.hedge_beta_usd = None
        

    # calcular beta
    # son 3, el de nvida, de applt y msft
    def compute_betas(self):
        self.posicion_beta = compute_beta(self.benchmark, self.position_security)
        self.posicion_beta_usd = self.posicion_delta_usd * self.posicion_beta
        
        for security in self.hedge_securitiess:
            self.hedge_betas.append(compute_beta(self.benchmark, security))
            
    
    def compute_optimal_hedge(self):
        # computa cálculos usando algebra matricial
        dimension = len(self.hedge_securitiess)
        if dimension != 2:
            print("----------")
            print("al no ser cuadrada, la matriz no tiene solución")
            return
        
        # matriz cuadrada con todos elementos 1's
        deltas = np.ones([dimension])
        
        
        matriz_t = np.transpose(np.column_stack([deltas, self.hedge_betas]))
        
        # da la cantidad de usd en delta y en beta
        targets = -np.array([[self.posicion_delta_usd ], [self.posicion_beta_usd]])
        
        # invierte la matriz_t y hace un producto matricial con la matriz de resultados
        #   Recordar que los resultados son los pesos que deben tenere las variables para que se dé la ecuacion
        self.hedge_weights = np.linalg.inv(matriz_t).dot(targets)
        
        # suma las ponderaciones ( son la suma de S1 y S2)
        self.hedge_delta_usd = np.sum(self.hedge_weights)
        
        # producto matricial, de la traspuesta de los betas X las ponderaciones optimas
        self.hedge_beta_usd = np.transpose(self.hedge_betas).dot(self.hedge_weights).item()
        
        print("")
        print(f"{self.position_security}, beta: {self.posicion_beta}")
                
        print("")
        print(f"{self.hedge_securitiess[0]}, beta: {self.hedge_betas[0]}")
        print(f"{self.hedge_securitiess[1]}, beta: {self.hedge_betas[1]}")
        
        print("")
        print(f"{self.hedge_securitiess[0]} optimal weight: {self.hedge_weights[0]}")
        print(f"{self.hedge_securitiess[1]} optimal weight: {self.hedge_weights[1]}")
        
        
    









