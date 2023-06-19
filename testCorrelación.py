
import matplotlib
import statsmodels.api as sm
import pandas as pd
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def diaLaborable(dia):
    if(dia.isoweekday()>6):
        lab=0 #Sábados y domingos
    else:
        lab=1
    if((dia.day,dia.month) in [(1,1),(6,1),(15,8),(25,12),(12,10),(1,11)]):
        lab=0
    if((dia.day,dia.month) in [(6,12),(8,12),(24,12), (31,12)]):
        lab=0
    return lab

def covid(dia):
    if(dia<datetime.strptime('2021-02-15',"%Y-%m-%d").date()):
        return 1
    if(dia >datetime.strptime('2021-12-15',"%Y-%m-%d").date() and dia<datetime.strptime('2022-01-15',"%Y-%m-%d").date()):
        return 1
    else:
        return 0

def replaceValues(data):
    data_reemplazado = data.replace(0, np.nan)
    for cambiodehora in ['2021-03-28 00:00:00', '2021-10-31 01:00:00', '2022-03-27- 01:00:00']:
        data_reemplazado.loc[cambiodehora]=None  
    data_reemplazado  = data_reemplazado.interpolate(method='linear')
    for nulos in ['2022-05-26']:
        data_reemplazado.loc[nulos]=data_reemplazado.loc['2022-05-25'].values
    return data_reemplazado


#Temperaturas
diccionarioPonderacionCCC = {'Oviedo':0., 'Madrid':0., 'Santander':0., 'Barcelona':0., 'Valencia':0., 'Alicante':0.,'Murcia':0.,'Sevilla':0., 'Zaragoza':0., 'España':0.}
dfTemperaturasHistoricas=pd.read_csv('Temperaturas-historico3.csv', header=0, index_col=0, sep=';',  decimal='.', encoding='UTF-8')
dfTemperaturasHistoricas.index=pd.to_datetime(dfTemperaturasHistoricas.index)

dfTempsMedia=dfTemperaturasHistoricas.groupby(dfTemperaturasHistoricas.index.date).mean().round(2)
dfTempsMedia.index=pd.to_datetime(dfTempsMedia.index)
dfTempsMedia['España']=np.round(dfTempsMedia.mean(axis=1),2)

def fillna_with_spain(row):
    if pd.isnull(row['España']):
        return row
    else:
        row.fillna(value=row['España'], inplace=True)
        return row

mejores_parametros={2:[(1,0,1),(2,1,0,7)], 3:[(1,0,1),(0,1,1,7)]}

df_3=pd.read_csv('ResB2C-historico.csv', header=0, index_col=0, sep=';',  decimal='.', encoding='UTF-8', dtype={'FechaLectura': 'str', 'Provincia': 'str', 'ccc': 'str', 'TFA': 'int', 'Wh': 'int', 'IdPS': 'int'})
df_agg=df_3.groupby(["FechaLectura","ccc","TFA"], as_index=False).aggregate({'Wh':'sum', 'IdPS':'sum'})
df_agg['diaLectura']=pd.to_datetime(df_3['FechaLectura']).apply(lambda x: x.date())
df_agg['Ratio']=df_agg['Wh']/df_agg['IdPS']

vTemperatura_medias=dict()
vTemperatura_medias['CCC']==np.round(dfTempsMedia[['Oviedo','Madrid', 'Sevilla', 'Valencia', 'Barcelona']].mean(axis=1),2)

vTemperaturas=dict()
def prepararTemperatura():
    global vTemperaturas
    #Temperaturas
    diccionarioPonderacionCCC = {'Oviedo':0., 'Madrid':0., 'Santander':0., 'Barcelona':0., 'Valencia':0., 'Alicante':0.,'Murcia':0.,'Sevilla':0., 'Zaragoza':0., 'España':0.}
    dfTemperaturasHistoricas=pd.read_csv('Temperaturas-historico3.csv', header=0, index_col=0, sep=';',  decimal='.', encoding='UTF-8')
    dfTemperaturasHistoricas.index=pd.to_datetime(dfTemperaturasHistoricas.index)
    dfTempsMedia=dfTemperaturasHistoricas.groupby(dfTemperaturasHistoricas.index.date).mean().round(2)
    dfTempsMedia.index=pd.to_datetime(dfTempsMedia.index)
    dfTempsMedia['España']=np.round(dfTempsMedia.mean(axis=1),2)
    # Aplicar la función a cada fila del DataFrame
    dfTempsMedia = dfTempsMedia.apply(fillna_with_spain, axis=1)
    vTemperaturas['CCC'] = np.round((dfTempsMedia[dfTempsMedia.columns.intersection(diccionarioPonderacionCCC.keys())] * pd.Series(diccionarioPonderacionCCC)).sum(axis=1),2)
    

def prediccionRatioARIMA_sinExog(y_train, y_truth, diasPrediccion, tarifa,ccc, order):
    #lab_train=vLaboralidad.loc[y_train.index]
    #lab_test=vLaboralidad.loc[y_truth.index]
    mejores_parametros= order#[order.order, order.seasonal_order]#[(1, 0, 1),(2, 1, 0, 7)] #SARIMAX(1, 0, 1)x(2, 1, [], 7)
    mod = sm.tsa.statespace.SARIMAX(
                                    y_train,
                                    order=mejores_parametros[0],
                                    seasonal_order=mejores_parametros[1],
                                    enforce_invertibility=False,
                                    )
    results = mod.fit()
    residuals = results.resid
    # Predicción a múltiples pasos
    pred_uc = results.get_forecast(steps=diasPrediccion)
    pred_ci = pred_uc.conf_int()
    ax = y[y_truth.index].plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Ratio')
    #actualSystem
    plt.legend()
    plt.title(' sin exog TFA:'+str(tarifa)+' ccc:'+ccc)
    plt.show()
    # Evaluación de la predicción ARIMA
    print('ARIMA sin regresores para '+str(tarifa)+ ' en comercializadora '+ ccc)
    print(order)
    predicciones_arima = pred_uc.predicted_mean[y_truth.index]
    mse = ((predicciones_arima - y_truth) ** 2).mean()
    rele = (np.abs(predicciones_arima - y_truth)/y_truth*100).mean()
    print('Error cuadrático medio ARIMA {}'.format(round(mse, 2)))
    print('Raíz cuadrada de ECM ARIMA {}'.format(round(np.sqrt(mse), 2))) 
    print('Error porcentual medio ARIMA {}'.format(round(np.sqrt(rele), 2))) 
    return residuals


def analizarCorrelacion(resid, variable):
    from scipy.stats import pearsonr, spearmanr
    #ind=pd.date_range(y_train.index[0],vTemperaturas[ccc].index[-1], closed='left')
    coef_pearson, p_value_pearson = pearsonr(resid, variable[resid.index])
    # coef_spearman, p_value_spearman = spearmanr(resid, variable[resid.index])
    return coef_pearson, p_value_pearson

ccc='CCC' 
df_agg_ccc=df_agg[df_agg['ccc']==ccc]
for tarifa in [2,3]:
    df_agg_tfa=df_agg_ccc[df_agg_ccc['TFA']==tarifa]
    #GUARDAR EL VALOR DE WH por tarifa real
    y_truthTFA=df_agg_tfa.groupby(["diaLectura"], as_index=False).aggregate({'Wh':'sum', 'IdPS':'sum'})
    y_truthTFA=y_truthTFA.set_index(y_truthTFA['diaLectura'])
    df_agg_tfa=df_agg_tfa.set_index(pd.to_datetime(df_agg_tfa['FechaLectura']))
    df_agg_tfa=df_agg_tfa.resample('H').aggregate({'Wh':'sum', 'IdPS':'sum'})
    df_agg_tfa['Wh']=replaceValues(df_agg_tfa['Wh'])
    df_agg_tfa['IdPS']=replaceValues(df_agg_tfa['IdPS'])
    df_agg_tfa['Ratio']=np.round(df_agg_tfa['Wh']/df_agg_tfa['IdPS'],2)
    dfRatio=df_agg_tfa.copy()      
    print(df_agg)
    dfRatio=dfRatio.resample('D').aggregate({'Wh':'sum', 'IdPS':'sum'})
    dfRatio['Ratio']=np.round(dfRatio['Wh']/dfRatio['IdPS'],2)
    y =dfRatio['Ratio']
    from pylab import rcParams
    rcParams['figure.figsize'] = 18, 8
    dias=pd.Series(index=y.index, data=y.index.values)
    vCovid=dias.copy().map(lambda x: covid(x.date()))
    vLaboralidad=dias.copy().map(lambda x: diaLaborable(x.date()))
    import pmdarima as pm
    comienzo={2:'2021-02-10', 3:'2021-09-01'}
    y_train = y.loc[comienzo[tarifa]:'2023-01-31']
    y_truth = y.loc['2023-02-01':'2023-02-11']
    diasPrediccion=len(y_truth)
    order=mejores_parametros[tarifa]#pmARIMA_sinExog(y_train)
    resid=prediccionRatioARIMA_sinExog(y_train, y_truth, diasPrediccion, tarifa, ccc, order)
    grafico=True
    if(grafico):
        plt.title('Posible correlación del residuo con la temperatura')
        plt.xlabel('Fecha')
        plt.ylabel('Temperatura (°C) y Residuo')
        vTemperaturas[ccc][resid.index].plot(label='Temperatura con ponderacion', color='r')
        vTemperatura_medias[ccc].loc[resid.index].plot(label='Temperatura media', color='y')
        resid.plot(label='Residuo', color='g')
        plt.legend()
        plt.show()
    print("--------------")
    coef_pearson, p_value_pearson=analizarCorrelacion(resid=resid['2022-02-17':], variable=vTemperaturas[ccc])
    print("Temperatura. coef_pearson: {}, p_value_pearson {}".format(coef_pearson, p_value_pearson))

    print("--------------")
    temperatura_umbral = np.mean(vTemperaturas[ccc])
    temperatura_cuadratica = (vTemperaturas[ccc] - temperatura_umbral) ** 2
    coef_pearson, p_value_pearson=analizarCorrelacion(resid=resid['2022-02-17':], variable=temperatura_cuadratica)
    print("Temperatura cuadrática. coef_pearson: {}, p_value_pearson {}".format(coef_pearson, p_value_pearson))
    temperatura_umbral = np.mean(vTemperaturas[ccc])
    temperatura_cuadratica = (vTemperaturas[ccc] -temperatura_umbral) ** 2 
    temperatura_cuadratica[y_train.index].plot()
    resid[y_train.index].plot()
    plt.show()