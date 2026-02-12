# -*- coding: utf-8 -*-




# imports

import numpy as np
#import pandas
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2
import importlib

import apoyo_var
importlib.reload(apoyo_var)


# inputs

degrees_fr = 20
size = 10**6
distribucion_tipo = "chi" 
# normal, student, exponencial, chi, uniforme


# Crea una instancia, con los inputs deseados
simulacion_1 = apoyo_var.simulador(distribucion_tipo, size, degrees_fr) 


# Genera el vector (con determinada distribución)
simulacion_1.generate_vector()  

# Computa los stats
simulacion_1.compute_stats()

# computa el test
simulacion_1.test_jb()

# computa los plots
simulacion_1.plot()

# asigna a una variable, el vector creado en la instancia
x = simulacion_1.vector









