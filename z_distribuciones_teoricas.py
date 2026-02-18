# -*- coding: utf-8 -*-

# imports

import numpy as np
#import pandas
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2
import importlib

import z_apoyo_var
importlib.reload(z_apoyo_var)

#################
# inputs
#################

# degrees_fr = 2
#size = 10**6

distribucion_tipo = "gamma" 
# normal estandart, normal, student, exponencial, chi, uniforme, gamma


# aca se generará el input, en una instancia de clase input
inputs = z_apoyo_var.sim_input()

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
simulacion_1 = z_apoyo_var.simulador(inputs) 


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









