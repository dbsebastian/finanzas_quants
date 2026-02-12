# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 16:54:56 2026

@author: Sebastian
"""


# imports

import numpy as np
#import pandas
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2
import importlib

import apoyo_var
importlib.reload(apoyo_var)


###################
# loop para JB test
###################


# inputs

degrees_fr = 20
size = 10**6
distribucion_tipo = "chi" 
# normal, student, exponencial, chi, uniforme


# params
n = 0
is_norm = True
str_title = "normal"
signif = 95


# calculos

signif_porcien = np.round((1 - signif/100), 3)

total_false = 0

while is_norm and n < 100:
    x = np.random.standard_normal(size=10**6)
    mu = x.mean()
    sd = x.std()
    skw = skew(x)
    kurts = kurtosis(x)
    lado_izq_test = skw**2
    lado_der_test = (1/4) * ( kurts **2 )

    test_jb = (size/6) * (lado_der_test + lado_izq_test)

    p_value = 1 - chi2.cdf(test_jb, df=2)
    is_normal = (p_value > signif_porcien)
    
    print(f"n= {n} | is_normal= {is_normal}")
    
    if is_normal == False:
        total_false += 1 
    else:
        pass
    n += 1

# proporción

print(f"La proporción de false son {total_false/500}")



# plot y title

str_title = (f"Distribución tipo {str_title}")

std_title = f' std = {str(np.round(sd, 2))}'

str_title += f'\n con media = {str(np.round(mu, 2))}  ; {std_title}'

kurt_title = f' kurtosis = {str(np.round(kurts, 2))}'

str_title += f'\n skew = {str(np.round(skw, 2))}  ; {kurt_title}'



str_title += f"\n y con un P-Valor del Test JB de {np.round(p_value, 4)}," 
str_title += f"\n Resultando en una distribución Normal, si o no? : {is_normal}"



plt.hist(x, color="g", alpha=0.5, bins=100)
plt.title(str_title)
plt.show()
