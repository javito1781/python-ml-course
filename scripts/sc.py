
#%% [markdown]
# # Carga de datos a través de la función read_csv

#En este apartado se van a detallar las formas en que se pueden leer los csv y en general documentos que contengan los datos de estudio y como se deben cargar en un dataframe.

#%%
import pandas as pd
data = pd.read_csv('C:/Workspace/mis_cursos/ML/python-ml-course/datasets/titanic/titanic3.csv', sep=',')
print(data)


#%% [markdown]
# # Carga de datos a través de la función open

#Lee los datos línea a línea a través de un bucle for y 
# borra los datos de la memoria una vez utilizados. Es mucho mas eficiente.
#%%
data2 = open('C:/Workspace/mis_cursos/ML/python-ml-course/datasets/titanic/titanic3.csv', 'r')
cols = data2.readline().strip().split(",")
print (str(cols)+'\n')
n_cols = len(cols)
print(n_cols)

#%% [markdown]
#Una vez conocemos las columnas y el número de ellas, creamos
# un diccionario que almacene las mismas
#%%
counter = 0
main_dict = {}
for col in cols:
    main_dict[col]=[]

for line in data2:
    values = line.strip().split(',')
    for i in range(len(cols)):
        main_dict[cols[i]].append(values[i])
    counter +=1
print ('El data set tiene %d filas y %d columnas\n'%(counter, n_cols))   
print (main_dict)

#%% [markdown]
#Ahora se puede crear un dataframe mediante la librería Pandas, haciendo uso
# del diccionario que se ha creado

#%%
df2 = pd.DataFrame(main_dict)
print(df2)


#%% [markdown]
# ## Lectura y escritura de ficheros

#%%
infile ="C:/Workspace/mis_cursos/ML/python-ml-course/datasets/customer-churn-model/Customer Churn Model.txt"
outfile ="C:/Workspace/mis_cursos/ML/python-ml-course/datasets/customer-churn-model/Tab Customer Churn Model.txt"


#%%
with open(infile, "r") as infile1:
    with open(outfile, "w") as outfile1:
        for line in infile1:
            fields = line.strip().split(",")
            outfile1.write("\t".join(fields))
            outfile1.write("\n")


#%%
df4 = pd.read_csv(outfile, sep = "\t")
df4.head()

#%% [markdown]
# # Leer datos desde una URL

#%%
medals_url = "http://winterolympicsmedals.com/medals.csv"


#%%
medals_data = pd.read_csv(medals_url)


#%%
medals_data.head()

import urllib3
http = urllib3.PoolManager()
r = http.request('GET', medals_url)
print(r.status)
response = r.data
print(response)

import csv
cr=csv.reader(response)