import pandas as pd
from datetime import datetime, timedelta
import numpy as np
dfResumen=pd.read_csv('ResB2C-historico.csv', header=0, index_col=0, sep=';',  decimal='.', encoding='UTF-8', dtype={'FechaLectura': 'str', 'Provincia': 'str', 'ccc': 'str', 'TFA': 'str', 'Wh': 'int', 'IdPS': 'int'})
####################
df_agg=dfResumen#.groupby(["FechaLectura","ccc","TFA"], as_index=False).aggregate({'Wh':'sum', 'IdPS':'sum'})
df_agg['Ratio']=df_agg['Wh']/df_agg['IdPS']

#filtros para reducir el dataset
agg_df=agg_df[agg_df['ccc']=='CCC']
agg_df=agg_df[agg_df['TFA']=='2']
# Resetamos el índice
agg_df = agg_df.reset_index(drop=True)

########################
agg_df['FechaLectura']=pd.to_datetime(agg_df['FechaLectura'], format='%Y-%m-%d %H:%M:%S')
agg_df=agg_df.drop(['TFA'], axis=1)
df_horas = agg_df.groupby([agg_df['FechaLectura'].dt.date, agg_df['FechaLectura'].dt.hour]).sum()
#df_horas = agg_df.groupby([agg_df['Provincia'],agg_df['FechaLectura'].dt.date, agg_df['FechaLectura'].dt.hour]).sum()
df_horas['Ratio']=np.round(df_horas['Wh']/df_horas['IdPS'],3)
def normalize_row(row):
    row_sum = row.sum()
    return row / row_sum

df_horas = df_horas.unstack()
df_norm = df_horas.Ratio.apply(normalize_row, axis=1)
                                        ##########################
                                        ########CLUSTERING########
                                        ##########################

import warnings
warnings.filterwarnings('ignore')
print('----------------------')
print('Media de cada variable')
print('----------------------')
df_norm.mean(axis=0)

print('-------------------------')
print('Varianza de cada variable')
print('-------------------------')
df_norm.var(axis=0)

print(df_norm)
df_norm=df_norm.dropna()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
X = np.array(df_norm)

from sklearn.manifold import TSNE
tsne = TSNE(2, random_state = 0)#perplexity = 9, early_exaggeration = 11, learning_rate = 451, random_state = 0
days_t = tsne.fit_transform(X) #_transform lo mismo
targets = [1, 2, 3, 4, 5, 6, 7]


import datetime
weekDays = ["Lunes","Martes","Miercoles","Jueves","Viernes","Sabado","Domingo"]
tipoDia=[weekDays[item.weekday()] for item in df_norm.index.values.tolist()]
principalDf = pd.DataFrame(data=days_t, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, pd.Series(tipoDia)], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('tSNE', fontsize = 20)
targets = weekDays
colors = ['r', 'g', 'b', 'c', 'm','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf.iloc[:,2] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
            , finalDf.loc[indicesToKeep, 'principal component 2']
            , c = color
            , s = 50)


ax.legend(targets)
if(True):
    for i, row in finalDf.iterrows():
        # Obtiene el valor del índice y lo convierte a cadena
        label = str(df_norm.index[i])
        # Obtiene las coordenadas x e y del punto
        x = row['principal component 1']
        y = row['principal component 2']
        # Dibuja la etiqueta en el gráfico
        ax.text(x, y, label, fontsize=7)
    plt.show()

import datetime
weekDays = ["Lunes","Martes","Miercoles","Jueves","Viernes","Sabado","Domingo"]
tipoDia=[weekDays[item.weekday()] for item in df_norm.index.values.tolist()]
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
print(principalComponents)


principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, pd.Series(tipoDia)], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)  
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = weekDays
colors = ['r', 'g', 'b', 'c', 'm','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf.iloc[:,2] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

ax.legend(targets)
print(pca.explained_variance_ratio_)
plt.show()

from umap import UMAP
reducer = UMAP(n_components=2)
embedding = reducer.fit_transform(X)
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title("Clustering utilizando UMAP")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()
ax.scatter(embedding[:, 0], embedding[:, 1])
principalDf = pd.DataFrame(data=embedding, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, pd.Series(tipoDia)], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)  
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('UMAP', fontsize = 20)
targets = weekDays
colors = ['r', 'g', 'b', 'c', 'm','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf.iloc[:,2] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

Grafico3D=False
if(Grafico3D):
    for i, row in finalDf.iterrows():
        # Obtiene el valor del índice y lo convierte a cadena
        label = str(df_norm.index[i])
        # Obtiene las coordenadas x e y del punto
        x = row['principal component 1']
        y = row['principal component 2']
        z = row['principal component 2']
        # Dibuja la etiqueta en el gráfico
        ax.text(x, y, z, label, fontsize=7)

ax.legend(targets)
plt.show()