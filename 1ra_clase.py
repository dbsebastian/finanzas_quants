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

#
simulacion_1 = apoyo_var.simulador(distribucion_tipo, size, degrees_fr)

simulacion_1.generate_vector()

x = simulacion_1.vector


# funciones

def estadisticos(var_x):
    mean_x = var_x.mean()
    median_x = np.median(var_x)
    std_x = var_x.std()
    kurt_x = kurtosis(var_x)
    skew_x = skew(var_x)
    
    return mean_x, median_x, std_x, skew_x, kurt_x



def distribucion(tipo_distribucion, size, df=2):
    """
    esta función crea una distribución según el tipo indicado 
    y del tamaño (size) indicado.
    
    ** Argumento opcional (df) que son los grados de libertad para chi2 y student
    
    Devuelve un array de tamaño (size) y de distribución (tipo_distribución)
    """
    
    if tipo_distribucion =="normal":
        x = np.random.standard_normal(size)
        
    elif tipo_distribucion == "student":
        x = np.random.standard_t(df=df, size=size)
        
    elif tipo_distribucion == "exponencial":
        x = np.random.exponential(scale=df, size=size )       

    elif tipo_distribucion == "chi":
        x = np.random.chisquare(df=df, size=size )

    else:
        x = np.random.uniform(low=-1 , high=1, size=size)
        
    return x





def titulo_graph(tipo_distrib, meanx, stdx, skewx, kurtx, p_valor, es_normal, signif):
    """
    Crea el título del graph, en base al tipo de distribución creada.
    
    Devuelve un str

    """
    str_title = ""
    
    if tipo_distrib == "normal":
        str_title = (f"Distribución tipo {distribucion_tipo}")
    
    elif tipo_distrib == "student":
        str_title = (f"Distribución tipo {distribucion_tipo} con {degrees_fr} grados de libertad")
    
    elif tipo_distrib == "exponencial":
        str_title = (f"Distribución tipo {distribucion_tipo} con coeficiente de {degrees_fr}")
        
    elif tipo_distrib == "chi":
        str_title = (f"Distribución tipo {distribucion_tipo}-cuadrado con {degrees_fr} grados de libertad")
        
    else:
        str_title = (f"Distribución tipo {distribucion_tipo}")
    
    
    std_title = f' std = {str(np.round(stdx, 2))}'
    
    str_title += f'\n con media = {str(np.round(meanx, 2))}  ; {std_title}'
    
    kurt_title = f' kurtosis = {str(np.round(kurtx, 2))}'
    
    str_title += f'\n skew = {str(np.round(skewx, 2))}  ; {kurt_title}'
    
    str_title += f"\n y con un P-Valor del Test JB de {np.round(p_valor, 4)},"
    
    str_title += f"\n dada una significancia de {signif}, la distribución es: {es_normal}"
        
    return str_title



def test_jb_normalidad(size, skew, kurtosis, signif):
    # test de normalidad JB

    lado_izq = skew**2
    lado_der = (1/4) * ( kurtosis **2 )

    test_jb = (size/6) * (lado_der + lado_izq)  # acá tengo el valor JB

    # ahora al valor de JB debo obtener su CHI2
    chi_test = chi2.cdf(test_jb, df=2) 
    # función de probabilidad acumultativa a la izq del punto obtenido

    # ahora se obtiene el p_valor
    p_valor = 1 - chi_test
    
    # significancia al tanto por uno
    signif = signif/100

    es_normal = (p_valor > signif)

    if es_normal:
        es_normal = "Normal"
    else:
        es_normal = "no Normal"
    
    return p_valor, es_normal, signif




# distribucion
#x = distribucion(distribucion_tipo, size=size, df=degrees_fr)

# estadisticos
mean_x, median_x, std_x, skew_x, kurt_x = estadisticos(x)

# test
p_value, is_normal, signif = test_jb_normalidad(size, skew_x, kurt_x, 5)

# titulo
str_title = titulo_graph(distribucion_tipo, mean_x, std_x, skew_x, kurt_x, p_value, is_normal, signif)





# plot

plt.hist(x, color="g", alpha=0.5, bins=100)

plt.title(str_title)

plt.axvline(x=mean_x, label="mean", color="r")

plt.axvline(x=median_x, label="median", color="b")

plt.show()


raise "aese"








###################
# loop para JB test
###################



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
