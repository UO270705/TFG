import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Función para reemplazar valores
def replaceValues(data):
    data_reemplazado = data.replace(0, np.nan)
    for cambiodehora in ['2021-03-28 00:00:00', '2021-10-31 01:00:00', '2022-03-27- 01:00:00']:
        data_reemplazado.loc[cambiodehora]=None  
    data_reemplazado  = data_reemplazado.interpolate(method='time')
    for nulos in ['2022-05-26']:
        data_reemplazado.loc[nulos]=data_reemplazado.loc['2022-05-25'].values
    return data_reemplazado

#Generación de una imagen de ruido blanco 
plt.title("Generación de ruido blanco")
plt.plot(np.random.normal(0, 1, size=1000))
plt.show()


#Cargar el conjunto de datos y filtrar por comercializadora
data=pd.read_csv('ResB2C-historico.csv', header=0, sep=';', decimal='.',encoding='utf-8', index_col=0, dtype={'ccc': 'str', 'diaLectura': 'str', 'TFA': 'int', 'Wh': 'float', 'IdPS': 'int', 'Ratio':'float'})
data['FechaLectura']=pd.to_datetime(data['FechaLectura'], format='%Y-%m-%d %H:%M:%S')
df_agg_ccc=data[data['ccc']=='HC30']
df_agg_tfa=df_agg_ccc[df_agg_ccc['TFA']==2]
df_horas = df_agg_tfa
df_horas=df_horas.set_index('FechaLectura')
df_horas=df_horas['Ratio'].resample('H').sum()

#Descomposiciones estacionales sin reemplazar los valores anómalos
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_data = seasonal_decompose(df_horas, model="additive")
decompose_data.plot()
plt.show()

#Descomposiciones estacionales reemplazar los valores anómalos
decompose_data = seasonal_decompose(replaceValues(df_horas), model="additive")
decompose_data.plot()
plt.show()

#Agregación de la serie de forma diaria
df_agg_dias=df_agg_tfa.set_index('FechaLectura')
df_agg_dias=df_agg_dias.resample('D').aggregate({'Wh':'sum', 'IdPS':'sum'})
df_agg_dias['Wh']=replaceValues(df_agg_dias['Wh'])
df_agg_dias['IdPS']=replaceValues(df_agg_dias['IdPS'])
df_agg_dias['Ratio']=df_agg_dias['Wh']/df_agg_dias['IdPS']
decompose_data = seasonal_decompose(df_agg_dias['Ratio'], model="additive")
decompose_data.plot()
plt.show()

#Estacionalidad estacional
seasonality=decompose_data.seasonal
seasonality.plot(color='green')
plt.show()

#Test de ADF
from statsmodels.tsa.stattools import adfuller
def test_estacionariedad(serie):
    # Realizar el test de Dickey-Fuller Aumentado
    resultado = adfuller(serie)
    # Obtener los valores críticos
    valores_criticos = resultado[4]
    print('Estadístico de prueba:', np.round(resultado[0],2))
    print('Valor p:', np.round(resultado[1],2))
    print('Valores críticos:')
    for key, value in valores_criticos.items():
        print(f'  {key}: {np.round(value,2)}')
    # Comprobamos si la serie es estacionaria o no
    if resultado[0] < valores_criticos['5%'] and resultado[1] < 0.05:
        print('La serie es estacionaria')
    else:
        print('La serie no es estacionaria')

print('Test adfuller Ratio por horas')
dftest = adfuller(replaceValues(df_horas), autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)

print('Test adfuller Ratio por días')
test_estacionariedad(df_agg_dias['Ratio'])
print('Test adfuller Ratio por días obviando el Covid')
test_estacionariedad(df_agg_dias['Ratio'].loc['2021-02-01':'2022-02-01'])

#Analizar la estacionareidad gráfica
rolling_mean = df_agg_dias['Ratio'].diff().rolling(window = 20).mean().dropna()
rolling_std = df_agg_dias['Ratio'].diff().rolling(window = 20).std().dropna()
df_agg_dias['Ratio'].diff().plot()
rolling_mean.plot(label='Media')
rolling_std.plot(label='Desviación típica')
plt.title('Análisis de estacionariedad')
plt.legend()
plt.show()

#Reemplazar la duración de la ventana en el que se calculan las diferentes métricas
rolling_mean = df_agg_dias['Ratio'].rolling(window = 7).mean().dropna()
data_rolling_mean_diff = rolling_mean - rolling_mean.shift()
ax1 = plt.subplot()
plt.show()
data_rolling_mean_diff.plot(legend='after rolling mean & differencing')
ax2 = plt.subplot()
df_agg_dias['Ratio'].plot(legend='original')
plt.legend()
plt.show()

#ADF de la serie diferenciada
dftest = adfuller(data_rolling_mean_diff.dropna(), autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
  print("\t",key, ": ", val)
