# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 17:38:33 2026

@author: Sebastian
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats



def load_timeseries(tick):
    
    # ^SPX, ^SPY
    # XLK (sector tecnológico)
    # XLV (salud)
    # XLF (Financiero)
    
    path = "h:\\Cursos\\Quants_yt\\Datos"
    path += "\\" + tick + ".csv"

    
    ## Pandas
    raw_data= pd.read_csv(path)
    t = pd.DataFrame()
    t["date"]= pd.to_datetime(raw_data["Date"], format="mixed", dayfirst=True)
    t["close"]= raw_data["Close"]
    t = t.sort_values(by="date", ascending=True)
    t["close_previous"]= t["close"].shift(1)
    t["return_close"]= t["close"]/t["close_previous"] - 1
    t= t.dropna()
    t = t.reset_index(drop=True)
    
    return t





class distribution:    
    # constructor

    def __init__(self, tick, significancia=0.05, decimals=5):
        
        self.tick = tick
        self.sign = significancia
        self.decimals = decimals
        self.str_title = None
        self.timeseries = None
        self.vector = None
        self.vector_returns = None
        
        # self.mean = None
        # self.median = None
        self.std = None
        self.skew = None
        self.kurt = None
        
        self.var_95 = None
        
        self.mean_anual = None
        self.std_anual = None
        self.sharpe_ratio = None
        
        self.test_jb = None
        self.chi_test = None
        self.es_normal = None
    
    
    def load_timeseries(self):
        
        self.timeseries = load_timeseries(self.tick)
        
        self.vector = self.timeseries["return_close"].values
        self.size = len(self.vector)
        self.str_title = f"{self.tick} | Real Data"
    
    
    def plot_timeseries(self):
        
        self.timeseries.plot(kind="line", x="date", y="close",
                             color='blue', 
                             alpha=0.8,
                             linestyle='--',
                             grid=True,
                             title=f'Timeserie de precios de Cierre de {self.tick}',
                )
        
        plt.show()
    
    
    # vemos
    def compute_stats(self, factor = 252):
        """
        al anualizar, se utilizan por defecto factor= 252 días.
        """
        self.mean = stats.tmean(self.vector)
        self.mean_anual = stats.tmean(self.vector) * factor
        self.median = np.median(self.vector)
        self.std_anual = stats.tstd(self.vector) * np.sqrt(factor)
        self.skew = stats.skew(self.vector)
        self.kurt = stats.kurtosis(self.vector)
        
        # Sharpe ratio
        self.sharpe_ratio = self.mean_anual / self.std_anual if self.std_anual > 0 else 0.0
        
        # valor en Riesgo
        self.var_95 = np.percentile(self.vector, 5)
        self.evar_95 = np.mean(self.vector[self.vector <= self.var_95])
        
        # obtiene el valor JB
        self.test_jb = (self.size/6)*((1/4)*( self.kurt **2 )+ self.skew**2)  

        # ahora se obtiene el p_valor   
        # ahora al valor de JB debo obtener su CHI2
        # función de probabilidad acumultativa a la izq del punto obtenido
        self.p_valor = 1 - ( stats.chi2.cdf(self.test_jb, df=2))
        self.es_normal = (self.p_valor > self.sign)

        if self.es_normal:
            self.es_normal = "Normal"
        else:
            self.es_normal = "no Normal"
        


    def plot_histogram(self):
        """
        Crea el título del graph, en base al tipo de distribución creada.
        """
        self.str_title = f"{self.tick}"
        
        self.str_title += f'\n Media anual: {str(np.round(self.mean_anual, self.decimals))}  |  SD anual: {str(np.round(self.std_anual, self.decimals))}'
        
        self.str_title += f"\n un Bias: {np.round(self.skew, self.decimals)} |  una Kurtosis de {np.round(self.kurt, self.decimals)}"
        
        self.str_title += f"\n un JB test: {np.round(self.test_jb, self.decimals)}, un P-Valor: {np.round(self.p_valor, self.decimals)},"
        
        self.str_title += f"\n la distribución es: {self.es_normal}"
        
        self.str_title += f"\n con un Ratio de Sharpe de {np.round(self.sharpe_ratio, self.decimals)}, y un Valor Riesgo de {np.round(self.var_95, self.decimals)}"

        plt.hist(self.vector, color="g", alpha=0.5, bins=100)
        plt.title(self.str_title)
        plt.axvline(x=self.mean, label="mean", color="r", alpha=0.5)
        plt.axvline(x=self.median, label="median", color="b", alpha=0.5)
        plt.axvline(x=self.var_95, label="valor Riesgo", color="#FF00FF")
        plt.show()



    




def computar_media_return_y_volatilidad(tick):
    
    """
    
    """
    decimals = 6
    
    t = load_timeseries(tick)
    
    x = t["return_close"].values
    
    average_return = np.round(np.meann(x), decimals)
    
    volatilidad = np.round(np.std(x, ddof=0), decimals)

    return average_return, volatilidad        



