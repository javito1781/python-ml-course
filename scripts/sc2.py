#%% [markdown]
# # 1. Resumen de los datos: dimensiones y estructuras

#%%
import pandas as pd
import os

filename = 'C:/Workspace/mis_cursos/ML/python-ml-course/datasets/titanic/titanic3.csv'
data = pd.read_csv(filename)
print(data.head(10))
print('\n')

data.shape

print(data.columns.values)
#%% [markdown]
# ## Resumen de los estadísticos y tipos de datos
# Una vez conseguidas las columnas del dataframe, vamos a calcular
# algunos estadísticos de las columnas del dataframe.

#%%
print(data.describe())

#%% [markdown]
# Tambien se pueden conocer los tipos de datos que componen cada una de
# de las columnas del dataframe mediante la siguiente sentencia.

#%%
print(data.dtypes)
print('\n')
#%% [markdown]
# # 2. Valores perdidos
# Al hacer la función describe, se puden obtener algunos valores estadísticos 
# del dataframe. Si por ejemplo se viera que un count que representa el número de valores de una
# columna es menor que el tamaño de la columna, significa que algún valor se ha perdido o corrompido.
#                                                                                                    
# No tratar este tipo de situaciones, puede llevar a un análisis erroneo de las situaciones y de los
# datos de los que disponemos.

#%%
nulls_body = pd.isnull(data['body'])
print(nulls_body)
print('\n')
print('\n')
#%%
notnulls_body = pd.notnull(data["body"])
print(notnulls_body)

#%% [markdown]
# Con estas dos funciones podemos saber cuantos valores nulos hay o cuantos valores no nulos hay, dependiendo
# de lo que interese. Para contar los nulos se pude utilizar la función ravel, la cual genera un array con los valores true
# false que obtenemos. Hecho eso, como para el lenguaje máquina true es 1 y false es 0, haciendo la suma de los valores,
# nos dira cuentos true (número de valores null) hay en el data set.

#%%
num_nulls = pd.isnull(data['body']).values.ravel().sum()
print (num_nulls)

#%%
num_not_nulls = pd.notnull(data['body']).values.ravel().sum()
print (num_not_nulls)

#%% [markdown]
# ## ¿Por qué faltan valores en un data set?
# Los valores de un data set pueden estar corrompidos y es importante conocer las razones por las que esto puede ocurrir.
# * Extracción de los datos de una base de datos
# * Recoleccion de los datos

#%% [markdown]
# ### Borrado de los valores que faltan
# En caso de no tener un 1% de los datos, borrar la columna entera no es una opción porque tenemos un 99% de los datos.
# En estas situaciones es interesante borrar solo aquellos valores que son nulos.
#%% [markdown]
# Si queremos borrar solo aquellas filas en las que todos los valores sean null, haremos lo siguiente.

#%%
data.dropna(axis=0, how='all')
print(data)
#%% [markdown]
# Si queremos borrar solo aquellas filas en las que alguno de los valores sean null, haremos lo siguiente.
#%%
data2 = data
data2 = data2.dropna(axis=0, how='any')
print(data2)

#%% [markdown]
# Como toda fila tiene algún campo con un valor nulo, mo borra todo.

#%% [markdown]
# ### Imputación de los valores faltantes
# El objetivo de este método, es sustituir los valores nulos por un valor numerico por ejemplo 0. Esto en ocasiones no tiene mucho sentido.

#%%
data3 = data
data3 = data3.fillna(0)
print(data3)

#%% [markdown]
# Otra forma es sustituir los valores nulos por una palabra.
#%%
data4 = data
data4 = data4.fillna("Desconocido")
print(data4)

#%% [markdown]
# La opción mas utilizada es la de sustituir cada null con un tipo de dato, en función de la columna

#%%
data5=data
data5['body'] = data5['body'].fillna(0)
data5["home.dest"] = data5["home.dest"].fillna('Desconocido')
print(data5)

#%% [markdown]
# Otra manera es sustituir los valores nulos por la media de los valores de la columna

#%%
num_nulls5 = pd.isnull(data5['age']).values.ravel().sum()
print(str(num_nulls5)+'\n')
data5['age'] = data5['age'].fillna(data5['age'].mean())
print (data5['age'])

#%% [markdown]
# Mediante esta metodología nos dan valores decimales que al hablar de la edad de una persona no tienen sentido. Hay un metodo que es ffill y backfill
# que sustituye los valores null por el valor anterior y posterior respectivamente

#%%
data6=data
data6['age'] = data6['age'].fillna(method='ffill')
print(data5['age'])

#%%
data6['age'] = data6['age'].fillna(method='backfill')
print(data5['age'])
#%% [markdown]
# # 3. Crear variables dummy
# Mediante este proceso se crean unas variables separadas para cada tipo de variable que se presente en los datos de estudio.
# Las variables categoricas son muy útiles de por si, pero en ocasiones para ver cierta causalidad en los datos, es conveniente
# separarlas. Un ejemplo de esto es la variable sex que contiene male y female en el dataset del titanic.

#%%
data = pd.read_csv(filename)
sex = data['sex']
print(sex)
#%%
dummy_sex=pd.get_dummies(data['sex'], prefix='sex')
print(dummy_sex)

#%%
column_name=data.columns.values.tolist()
column_name

data = data.drop(["sex"], axis = 1)
data = pd.concat([data, dummy_sex], axis = 1)

print (data)

#%% [markdown]
# # 4. Plots y visualización de los datos
# En esta última parte del tema de preparación de los datos, se van a mostrar los diferentes maneras de hacer una representación
# correcta de los datos. Esto ayuda a un análisis preliminar de los datos.

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('C:/Workspace/mis_cursos/ML/python-ml-course/datasets/customer-churn-model/Customer Churn Model.txt')
print(data)

#%%
#get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ### Scatter Plot

#%%
plot1 = data.plot(kind='scatter', x="Day Mins", y="Day Charge")
print(plot1)
#%%
plot2 = data.plot(kind='scatter', x="Night Mins", y="Night Charge")
print(plot2)

#%%
figure, axs = plt.subplots(2,2, sharey=True, sharex =True)
data.plot(kind='scatter', x="Day Mins", y="Day Charge", ax=axs[0][0])
data.plot(kind='scatter', x="Night Mins", y="Night Charge", ax=axs[0][1])
data.plot(kind='scatter', x="Day Calls", y="Day Charge", ax=axs[1][0])
data.plot(kind='scatter', x="Night Calls", y="Night Charge", ax=axs[1][1])

print(figure)

#%% [markdown]
# ### Histograma de frecuencias


#%%
k=int(np.ceil(1+np.log2(3333)))
plt.hist(data['Day Calls'], bins=k)
plt.xlabel('Numero de llamadas al dia')
plt.ylabel('Frecuencia')


#%% [markdown]
# ### Boxplot, diagrama de caja y bigotes

#%%
plt.boxplot(data['Day Calls'])
plt.ylabel('Numero de llamadas diarias')
plt.title('Boxplot de las llamadas diarias')

#%%
stat = data["Day Calls"].describe()
print(stat)