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

import scipy.optimize as opt

import sop_market_data
importlib.reload(sop_market_data)


def compute_beta(benchmark, security):
    m = model(benchmark, security)
    m.sync_timeseries()
    m.compute_regress()
    return m.beta


def compute_corr(security_1, security_2):
    m = model(security_1, security_2)
    m.sync_timeseries()
    m.compute_regress()
    return m.r_square


def dataframe_corr_beta(benchmark, position_security, hedge_universe):
    decimals = 5
    df = pd.DataFrame()
    correlaciones = list()
    betas = list()
    for hedge_security in hedge_universe:
        corr = compute_corr(position_security, hedge_security)
        correlaciones.append(np.round(corr, decimals))
        
        betas_score = compute_beta(benchmark, hedge_security)
        betas.append(np.round(betas_score , decimals))
        
    df["hedge_security"] = hedge_universe
    df["correlations"] = correlaciones
    df["betas"] = betas
    df = df.sort_values(by="correlations", ascending=False)
    return df

# define la función a minimizar
def cost_function_capm(x, betas, target_delta, target_beta, regularization):
    dimensiones = len(x) # q' de dimensiones
    # vector de 1s
    deltas = np.ones([dimensiones])
    # Crea la función delta, (parte del sistema de ecuación a optimizar) 
    f_delta = (np.transpose(deltas).dot(x).item() + target_delta)**2
    # crea la función beta, parte del sistema de ecuación a optimizar
    f_beta = (np.transpose(betas).dot(x) + target_beta)**2
    # por ahora sin función de regularización
    f_penalty = regularization*(np.sum(x**2))    
    f = f_delta + f_beta + f_penalty
    
    return f


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
        if dimension < 2:
            print("----------")
            print("Debe ser de al menos 2 dimensiones")
            return
        
        # matriz cuadrada con todos elementos 1's
        deltas = np.ones([dimension])
        
        
        matriz_t = np.transpose(np.column_stack([deltas, self.hedge_betas]))
        
        # da la cantidad de usd en delta y en beta
        targets = -np.array([[self.posicion_delta_usd ], [self.posicion_beta_usd]])
        
        # invierte la matriz_t y hace un producto matricial con la matriz de resultados
        #   Recordar que los resultados son los pesos que deben tenere las variables para que se dé la ecuacion
        if dimension ==2:
            self.hedge_weights = np.linalg.inv(matriz_t).dot(targets)
            
        else:
            self.hedge_weights = np.linalg.pinv(matriz_t).dot(targets)
        
        # suma las ponderaciones ( son la suma de S1 y S2)
        self.hedge_delta_usd = np.sum(self.hedge_weights)
        
        # producto matricial, de la traspuesta de los betas X las ponderaciones optimas
        self.hedge_beta_usd = np.transpose(self.hedge_betas).dot(self.hedge_weights).item()
        
        # código que expone los valores obtenidos
        print("")
        print(f"{self.position_security}, beta: {self.posicion_beta}")
        
        #ticker_betas = list()
        #betas_valores = list()
        
        print()
        print("Los valores de beta originales:")
        print()
        for i in range(len(self.hedge_securitiess)):
            print("---")
            print(f"   {self.hedge_securitiess[i]} con beta: {self.hedge_betas[i]}")
            print()
        
        print()
        print("Los valores de beta minimos 'libres' óptimos:")
        print()
        for i in range(len(self.hedge_securitiess)):
            print("---")
            print(f"   {self.hedge_securitiess[i]} con beta: {self.hedge_weights[i]}")
            print()
                
        
        
    def compute_hedge_weights(self, reg_param = 0):   
        dimension = len(self.hedge_securitiess)
        # inputs
        # self.hedge_betas          →    son las betas de cada activo a usar como hedge
        # self.posicion_delta_usd   →    $$ de la posición
        # self.posicion_beta_usd    →    $$ en betas
                
        # Crea los valores de x, que serán usados para iterar la función
        # target delta, es la posición en dolares inicial $$.
        # al dividirla por la cantidad de betas, sabemos las "ponderación" aproximada de beta dolares en cada activo de hedge
        x0 = - self.posicion_delta_usd /len(self.hedge_betas) * np.ones([len(self.hedge_betas), 1])
        
        # computando la optimización
        # minimize toma la función
        # los posibles valores que tendrá el dominio se indican mediante x0
        # y los argumentos o variables a ser usadas en el sistema de ecuación se ingresan mediante args
        optimal_result = opt.minimize(fun=cost_function_capm,
                                      x0=x0.flatten(),
                                      args=(self.hedge_betas,           \
                                            self.posicion_delta_usd,    \
                                            self.posicion_beta_usd,     \
                                            reg_param )                 \
                                          )
        
        self.hedge_weights_optimizados = optimal_result.x
        # if dimension ==2:
        #     self.hedge_weights = np.linalg.inv(matriz_t).dot(targets)
            
        # else:
        #     self.hedge_weights = np.linalg.pinv(matriz_t).dot(targets)
        
        # suma las ponderaciones ( son la suma de S1 y S2)
        self.hedge_delta_usd = np.sum(self.hedge_weights)
        
        # producto matricial, de la traspuesta de los betas X las ponderaciones optimas
        self.hedge_beta_usd = np.transpose(self.hedge_betas).dot(self.hedge_weights).item()
        
        
        print()
        print()
        print(f"Los pesos de Hedge que minimizan son:")
        print(f"considerando una regularización de {reg_param}")
        for i in range(len(self.hedge_weights_optimizados)):
            print("---")
            print(f"     {self.hedge_securitiess[i]} {self.hedge_weights_optimizados[i]}")
            print()
        print("--")
        print(f"Dónde")
        print(f"- Posición beta de:  {self.posicion_beta_usd}")
        print(f"- Posición delta de: {self.posicion_delta_usd}")











