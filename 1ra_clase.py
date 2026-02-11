# -*- coding: utf-8 -*-




# imports

import numpy as np
#import pandas
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2


# inputs

degrees_fr = 2000
size = 10**6

distribucion_tipo = "chi" 
# normal, student, exponencial, chi, uniforme




# funciones

def estadisticos(var_x):
    
    mean_x = var_x.mean()
    
    median_x = np.median(var_x)
    
    std_x = var_x.std()
    
    kurt_x = kurtosis(var_x)
    
    skew_x = skew(var_x)
    
    
    return mean_x, median_x, std_x, skew_x, kurt_x



#  codigo

str_title = ""

if distribucion_tipo == "normal":
    
    x = np.random.standard_normal(size)
    
    mean_x, median_x, std_x, skew_x, kurt_x = estadisticos(x)
    
    str_title = (f"Distribución tipo {distribucion_tipo}")
    
    

elif distribucion_tipo == "student":

    x = np.random.standard_t(df=degrees_fr, size=size)
    
    mean_x, median_x, std_x, skew_x, kurt_x = estadisticos(x)
    
    str_title = (f"Distribución tipo {distribucion_tipo} con {degrees_fr} grados de libertad")
    
    

elif distribucion_tipo == "exponencial":
    
    x = np.random.exponential(scale=degrees_fr, size=size )
    
    mean_x, median_x, std_x, skew_x, kurt_x = estadisticos(x)
    
    str_title = (f"Distribución tipo {distribucion_tipo} con coeficiente de {degrees_fr}")
    
    

elif distribucion_tipo == "chi":
    
    x = np.random.chisquare(df=degrees_fr, size=size )
    
    mean_x, median_x, std_x, skew_x, kurt_x = estadisticos(x)
    
    str_title = (f"Distribución tipo {distribucion_tipo}-cuadrado con {degrees_fr} grados de libertad")
    
    

else:
    x = np.random.uniform(low=-1 , high=1, size=size)
    
    mean_x, median_x, std_x, skew_x, kurt_x = estadisticos(x)

    str_title = (f"Distribución tipo {distribucion_tipo}")
    
    


# title

std_title = f' std = {str(np.round(std_x, 2))}'

str_title += f'\n con media = {str(np.round(mean_x, 2))}  ; {std_title}'

kurt_title = f' kurtosis = {str(np.round(kurt_x, 2))}'

str_title += f'\n skew = {str(np.round(skew_x, 2))}  ; {kurt_title}'







# test de normalidad JB

lado_izq = skew_x**2
lado_der = (1/4) * ( kurt_x **2 )

test_jb = (size/6) * (lado_der + lado_izq)  # acá tengo el valor JB

# ahora al valor de JB debo obtener su CHI2
chi_test = chi2.cdf(test_jb, df=2) 
# función de probabilidad acumultativa a la izq del punto obtenido

# ahora se obtiene el p_valor
p_valor = 1 - chi_test

es_normal = (p_valor > 0.05)

if es_normal:
    es_normal = "Normal"
else:
    es_normal = "no Normal"


str_title += f"\n y con un P-Valor del Test JB de {np.round(p_valor, 4)}," 
str_title += f"\n resultando en una distribución: {es_normal}"


# plot

plt.hist(x, color="g", alpha=0.5, bins=100)

plt.title(str_title)

plt.axvline(x=mean_x, label="mean", color="r")

plt.axvline(x=median_x, label="median", color="b")




