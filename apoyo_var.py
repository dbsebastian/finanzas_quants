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
    
    # acá van los inputs: 
    def __init__(self, tipo_dist, size, degree_f):
        self.dist = tipo_dist
        self.size = size
        self.df = degree_f
        self.title = None
        self.vector = None
    
    
    def generate_vector(self):
        """
        esta función crea una distribución según el tipo indicado 
        y del tamaño (size) indicado.
        
        ** Argumento opcional (df) que son los grados de libertad para chi2 y student
        
        Devuelve un array de tamaño (size) y de distribución (tipo_distribución)
        """
        tipo_distribucion = self.dist
        size = self.size
        df = self.df
        
        if tipo_distribucion =="normal":
            self.vector = np.random.standard_normal(size)
            
        elif tipo_distribucion == "student":
            self.vector = np.random.standard_t(df=df, size=size)
            
        elif tipo_distribucion == "exponencial":
            self.vector = np.random.exponential(scale=df, size=size )       

        elif tipo_distribucion == "chi":
            self.vector = np.random.chisquare(df=df, size=size )

        else:
            self.vector = np.random.uniform(low=-1 , high=1, size=size)
            
