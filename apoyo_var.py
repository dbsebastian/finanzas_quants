# -*- coding: utf-8 -*-


"""
Created on Wed Feb 11 20:40:33 2026

@author: Sebastian

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import skew, kurtosis, chi2



class simulador():
    
    # constructor
    # sólo lo usaré para declarar 
    def __init__(self, tipo, size, degree_f, significancia=0.05):
        self.tipo = tipo
        self.size = size
        self.df = degree_f
        self.sign = significancia
        # self.str_title = None
        # self.vector = None
        
        # self.x_mean = None
        # self.x_median = None
        # self.x_std = None
        # self.x_skew = None
        # self.x_kurt = None
        
        # self.test_jb = None
        # self.chi_test = None
        # self.es_normal = None
        
        
    
    # vector
    def generate_vector(self):
        """
        esta función crea una distribución según el tipo indicado 
        y del tamaño (size) indicado.
        
        ** Argumento opcional (df) que son los grados de libertad para chi2 y student
        
        Devuelve un array de tamaño (size) y de distribución (tipo_distribución)
        """
        
        if self.tipo =="normal":
            self.vector = np.random.standard_normal(self.size)
            
        elif self.tipo == "student":
            self.vector = np.random.standard_t(df=self.df, size=self.size)
            
        elif self.tipo == "exponencial":
            self.vector = np.random.exponential(scale=self.df, size=self.size )       

        elif self.tipo == "chi":
            self.vector = np.random.chisquare(df=self.df, size=self.size )

        else:
            self.vector = np.random.uniform(low=-1 , high=1, size=self.size)
            
    
    # vemos
    def compute_stats(self):
        
        self.x_mean = self.vector.mean()
        self.x_median = np.median(self.vector)
        self.x_std = self.vector.std()
        self.x_skew = skew(self.vector)
        self.x_kurt = kurtosis(self.vector)


    def test_jb(self):
        
        # obtiene el valor JB
        self.test_jb = (self.size/6)*((1/4)*( self.x_kurt **2 )+ self.x_skew**2)  

        # ahora se obtiene el p_valor   
        # ahora al valor de JB debo obtener su CHI2
        # función de probabilidad acumultativa a la izq del punto obtenido
        self.p_valor = 1 - ( chi2.cdf(self.test_jb, df=2))
        self.es_normal = (self.p_valor > self.sign)

        if self.es_normal:
            self.es_normal = "Normal"
        else:
            self.es_normal = "no Normal"


    def plot(self):
        """
        Crea el título del graph, en base al tipo de distribución creada.
        """
        self.str_title = ""
        
        if self.tipo == "normal":
            self.str_title = (f"Distribución tipo {self.tipo}")
        
        elif self.tipo == "student":
            self.str_title = (f"Distribución tipo {self.tipo} con {self.df} grados de libertad")
        
        elif self.tipo == "exponencial":
            self.str_title = (f"Distribución tipo {self.tipo} con coeficiente de {self.df}")
            
        elif self.tipo == "chi":
            self.str_title = (f"Distribución tipo {self.tipo}-cuadrado con {self.df} grados de libertad")
            
        else:
            self.str_title = (f"Distribución tipo {self.tipo}")
        
        self.std_title = f' std = {str(np.round(self.x_std, 2))}'
        
        self.str_title += f'\n con media = {str(np.round(self.x_mean, 2))}  ; {self.std_title}'
        
        self.kurt_title = f' kurtosis = {str(np.round(self.x_kurt, 2))}'
        
        self.str_title += f'\n skew = {str(np.round(self.x_skew, 2))}  ; {self.kurt_title}'
        
        self.str_title += f"\n y con un P-Valor del Test JB de {np.round(self.p_valor, 4)},"
        
        self.str_title += f"\n dada una significancia de {self.sign}, la distribución es: {self.es_normal}"

        plt.hist(self.vector, color="g", alpha=0.5, bins=100)
        plt.title(self.str_title)
        plt.axvline(x=self.x_mean, label="mean", color="r")
        plt.axvline(x=self.x_median, label="median", color="b")  
        plt.show()






        



