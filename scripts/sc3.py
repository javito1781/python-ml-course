#%% [markdown]
# # 1. Data Wranging- La cirugía de los datos
# Se denomina Data Wranging a la disputa de datos, a veces conocida como munging de datos, es el proceso de transformación y mapeo de datos de un formulario de datos "sin procesar" en otro formato con la intención de hacerlo más apropiado y valioso para una variedad de propósitos.

#%%
import random
import numpy as np
import pandas as pd
data = pd.read_csv('C:/Workspace/mis_cursos/ML/python-ml-course/datasets/customer-churn-model/Customer Churn Model.txt')
print(data)

#%% [markdown]
# ### Crear un subconjunto de datos

#%%
account_length = data["Account Length"]
print(account_length)

#%% [markdown]
# Para hacer un subconjunto de datosse utiliza el siguiente método.

#%%
subset = data[['Account Length', 'Eve Charge', 'Day Calls']]
print(subset)

#%%
desired_columns = ['Account Length', 'Eve Charge', 'Day Calls']
subset = data[desired_columns]
print(subset)

#%%
desired_columns = ["Account Length", "VMail Message", "Day Calls"]
desired_columns
all_columns_list = data.columns.values.tolist()
all_columns_list
sublist = [x for x in all_columns_list if x not in desired_columns]
sublist
subset = data[sublist]
print(subset)

#%% [markdown]
# En los dataframes, se puede hacer un filtrado de los datos, de tal manera que nos quedemos con un determinado
# número de columnas. Además de esto, también se pueden filtrar las columnas en función de una condición, para lo cual
# hay que hacer uso de los operadores lógicos. De esta manera podemos crear un nuevo data set con los datos que nos
# interesen en función del campo de estudio y de los datos que tengamos.

#%%[markdown]
# Usuarios con Day Mins > 300
#%%
data1 = data[data["Day Mins"]>300]
data1.shape


#%%[markdown]
# Usuarios de Nueva York (State = "NY")
#%%
data2 = data[data["State"]=="NY"]
data2.shape


#%%
##AND -> &
data3 = data[(data["Day Mins"]>300) & (data["State"]=="NY")]
print(data3)
data3.shape


#%%
##OR -> |
data4 = data[(data["Day Mins"]>300) | (data["State"]=="NY")]
data4.shape

#%% [markdown]
# Al igual que se hacen filtrados de la columnas y filtrado de las filas, se pueden hacer filtrados de columnas y filas de manera simultaneamente.
# Ahora se van a filtrar los minustos de dia, de noche de la cuenta de los 50 primeros individuos, un filtrado de dos columnas y 50 filas.

#%%
subset_first_50 = data[['Day Mins', 'Night Mins', 'Account Length']][:50]
subset_first_50.shape
print(subset_first_50)

#%% [markdown]
# El método ix permite hacer una consulta por fila y columna de manera simultánea. Esto se puede ver a continuación. La sentencia sería
# dataframe.ix[filas, columnas]

#%%
data.ix[1:10, 3:6]

#%% [markdown]
# Aunque el filtrado lo hace correctamente, se puede ver que sale una alerta indicando que el paquete esta obsoleto. Esto es así porque ix se
# solía utilizar en python 2.7. En python 3.7 se debe utilizar loc e iloc que se muestran a continuación. La diferencia entre uno y otro es que loc
# filtra las columnas por el nombre de la misma e iloc filtra por el número de la columna.

#%%
data.iloc[1:10, 3:6]

#%%
data.iloc[1:10, [2,5,7]]

#%%
data.iloc[[1,5,8,36], [2,5,7]]

#%%
data.loc[[1,5,8,36], ['Area Code', 'Phone', 'Day Mins']]

#%%
data['Total Mins'] = data['Day Mins'] + data['Night Mins'] + data['Eve Mins']
data['Total Mins'].head


#%% [markdown]
# ## Generación de números aleatorios
# Los números aleatorios son números que se presupone que pueden tomar volores diferentes cada vez que se llama a la función que los genera. Esto no es
# exáctamente así, debido a que se utiliza una semilla. Esta semilla la utiliza el sistema para generar una función que dan como resultado una serie de números
# psudoaleatorios que se repiten. El tiempo que tarda en repetirse el patrón es tan alto que se dice que son aleatorios cuando en realidad no lo son.

#%%
np.random.randint(1,100)

#%%
np.random.random()

def randint_list(n, a, b):
    x = []
    for i in range(n):
        x.append(np.random.randint(a, b))
    return x

randint_list(25, 1, 50)

#%% [markdown]
# A través de la librería random, podemos generar números aleatorios dentro de un rango y condicionarlos, es decir, poner reglas para que esos números
# cumpla con una regla preestablecida, por ejemplo que sean múltiplos de un número, en este caso se indican que esten entre 0 y 100, y que sean múltimplos de 7.
#%%
for i in range(10):
    print(random.randrange(0, 100, 7))

#%% [markdown]
# ## Shuffling
# Mediante este método se pueden mezclar los valores obtenidos mediante una lista ordenada.

#%%
a = np.arange(100)
print(a)

#%% [markdown]
# Mezclamos a
#%%
np.random.shuffle(a)
print(a)

#%% [markdown]
# ## Choice
# Este método permite seleccionar de manera aleatoria columnas de un dataframe. Para ello lo primero
# que hay que hacer es generar una lista con los nombres de todas las columnas del dataframe y lanzar esta
# función que nos dará como resultado una columna concreta.

#%%
column_list = data.columns.values.tolist()
print (column_list)

np.random.choice(column_list)


#%% [markdown]
# ## Semilla de números aleatorios
# Para asegurar la repitibilidad del experimento que vamos a llevar a cabo, es importante establecer una semilla
# con la que el sistema pueda llevar a cabo la generación de números aleatorios. Esto es así porque al establecer la
# semilla, el sistema siempre generará los mismos números aleatorios.


#%%
np.random.seed(2018)
for i in range(5):
    print(np.random.random())

#%% [markdown]
# ## Funciones de probabilidad
# ### Bernoulli
# Las descripciones matemáticas y las distribuciones están en los apuntes, en esta parte se va a desarrollar un bloque,
# haciendo uso de la distribución en python.

#%%
from scipy.stats import bernoulli
import matplotlib.pyplot as plt 
p = 0.7
mean, var, skew, kurt = bernoulli.stats(p, moments='mvsk')
print("Media %f"%mean)
print('Variaza %f'% var)
print('Sesgo %f'%skew)
print('Kurtosis %f'%kurt)
fix, ax=plt.subplots(1,1)
x=bernoulli.rvs(p, size = 1000)
ax.hist(x)
plt.show

#%% [markdown]
# ### Binomial

#%%
from scipy.stats import binom
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1)
n = 7
p = 0.4
mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
print("Media %f"%mean)
print('Variaza %f'% var)
print('Sesgo %f'%skew)
print('Kurtosis %f'%kurt)
x = np.arange(binom.ppf(0.01,n,p), binom.ppf(0.9999, n,p))
ax.plot(x, binom.pmf(x, n, p), 'bo', ms = 8, label = 'Funcion de densidad de B(7,0.4)')
ax.vlines(x, 0, binom.pmf(x,n,p), colors = 'b', lw = 4, alpha=0.5)

rv = binom(n,p)
ax.vlines(x,0, rv.pmf(x), colors = 'k', linestyles = '--', lw=1, label='Distribución teórica')
ax.legend(loc='best', frameon = False) 
plt.show

#%% [markdown]
# ### Geométrica
#%%
from scipy.stats import geom
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1)
p=0.3
mean, var, skew, kurt = geom.stats(p, moments = 'mvsk')
print("Media %f"%mean)
print('Variaza %f'% var)
print('Sesgo %f'%skew)
print('Kurtosis %f'%kurt)

x = np.arange(geom.ppf(0.01,p), geom.ppf(0.99, p))
ax.plot(x, geom.pmf(x, p), 'bo', ms = 8, label = 'Distribución Geom(0.3')
ax.vlines(x,0,geom.pmf(x,p))
