import pandas as pd
import numpy as np
from datetime import datetime


def diaLaborable(dia):
    if(dia.isoweekday()>6):
        lab=1 #Sábados y domingos
    else:
        lab=0
    if((dia.day,dia.month) in [(1,1),(6,1),(15,8),(25,12),(12,10),(1,11)]):
        lab=1
    if((dia.day,dia.month) in [(6,12),(8,12),(24,12), (31,12)]):
        lab=2
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
diccionarioPonderacionCCC = {'Oviedo':0., 'Madrid':0.}#DISTRIBUCIÓN DE LA CARTERA DE CLIENTES POR PROVINCIAS
dfTemperaturasHistoricas=pd.read_csv('Temperaturas-historico3.csv', header=0, index_col=0, sep=';',  decimal='.', encoding='UTF-8')

def fillna_with_spain(row):
    if pd.isnull(row['España']):
        return row
    else:
        row.fillna(value=row['España'], inplace=True)
        return row


#Crear series temporales de temperaturas para utilizar como variable exógena
vTemperaturas=dict()
def prepararTemperatura():
    global vTemperaturas
    diccionarioPonderacionCCC = {'Oviedo':0., 'Madrid':0.}#, 'Santander':0., 'Barcelona':0., 'Valencia':0., 'Alicante':0.,'Murcia':0.,'Sevilla':0., 'Zaragoza':0., 'España':0.}
    dfTemperaturasHistoricas=pd.read_csv('Temperaturas-historico3.csv', header=0, index_col=0, sep=';',  decimal='.', encoding='UTF-8')
    dfTemperaturasHistoricas.index=pd.to_datetime(dfTemperaturasHistoricas.index)
    dfTempsMedia=dfTemperaturasHistoricas.groupby(dfTemperaturasHistoricas.index.date).mean().round(2)
    dfTempsMedia.index=pd.to_datetime(dfTempsMedia.index)
    dfTempsMedia['España']=np.round(dfTempsMedia.mean(axis=1),2)
    # Aplicar la función a cada fila del DataFrame
    dfTempsMedia = dfTempsMedia.apply(fillna_with_spain, axis=1)
    vTemperaturas['CCC'] = np.round((dfTempsMedia[dfTempsMedia.columns.intersection(diccionarioPonderacionCCC.keys())] * pd.Series(diccionarioPonderacionCCC)).sum(axis=1),2)

#Función para realizar predicciones y evaluar SARIMAX con distintas variables exógenas 
def prediccionRatioARIMA(y_train, y_truth, diasPrediccion, vLaboralidad, vTemperatura, vCovid, tarifa,ccc, order):
    vTemperatura.index=pd.to_datetime(vTemperatura.index)
    temperatura_train =vTemperatura.loc[y_train.index]
    temperatura_test =vTemperatura.loc[y_truth.index]
    """
    vTemperaturaMin.index=pd.to_datetime(vTemperaturaMin.index)
    vTemperaturaMin=vTemperaturaMin.resample('D').sum()
    temperaturaMin_train =vTemperaturaMin.loc[y_train.index]
    temperaturaMin_test =vTemperaturaMin.loc[y_truth.index]
    vTemperaturaMax.index=pd.to_datetime(vTemperaturaMax.index)
    vTemperaturaMax=vTemperaturaMax.resample('D').sum()
    temperaturaMax_train =vTemperaturaMax.loc[y_train.index]
    temperaturaMax_test =vTemperaturaMax.loc[y_truth.index]
    """
    covid_train=vCovid.loc[y_train.index]
    covid_test=vCovid.loc[y_truth.index] 
    vLaboralidad.index=pd.to_datetime(vLaboralidad.index)
    vLaboralidad=vLaboralidad.resample('D').sum()
    lab_train=vLaboralidad.loc[y_train.index]
    lab_test=vLaboralidad.loc[y_truth.index]
    mejores_parametros= order#[order.order, order.seasonal_order] #ARIMA(1,0,1)(0,1,1)[7]
    mod = sm.tsa.statespace.SARIMAX(
                        y_train, 
                        exog=pd.DataFrame({'temperatura': temperatura_train}),
                        order=mejores_parametros[0],
                        seasonal_order=mejores_parametros[1],
                        enforce_invertibility=False,
                        )
    results = mod.fit()
    # Predicción a múltiples pasos
    pred_uc = results.get_forecast(steps=diasPrediccion, exog=temperatura_test)
    pred_ci = pred_uc.conf_int()
    ax = y[y_truth.index].plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(label='Forecast')
    ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Ratio')
    plt.legend()
    plt.title(' TFA:'+str(tarifa)+' CCC:'+ccc)
    plt.show()
    # Evaluación de la predicción ARIMA
    print('ARIMA con regresores para TFA '+str(tarifa)+ ' en comercializadora '+ ccc)
    print(order)
    predicciones_arima = pred_uc.predicted_mean[y_truth.index]
    mse = ((predicciones_arima - y_truth) ** 2).mean()
    rele = (np.abs(predicciones_arima - y_truth)/y_truth*100).mean()
    print('Error cuadrático medio ARIMA {}'.format(round(mse, 2)))
    print('Raíz cuadrada de ECM ARIMA {}'.format(round(np.sqrt(mse), 1))) #reducido un grado
    print('Error porcentual medio ARIMA {}'.format(round(np.sqrt(rele), 1))) #reducido un grado
    return round(np.sqrt(rele), 1)


#Función para realizar predicciones y evaluar SARIMAX con variables exógenas con temperaturas transformadas
def prediccionRatioARIMA_tempCuadratica(y_train, y_truth, diasPrediccion, vLabor, vTemperatura, vTemperaturaMin, vTemperaturaMax, vCovid, tarifa,ccc, order):
    #Elección de las variables exógenas a utilizar
    vTemperatura.index=pd.to_datetime(vTemperatura.index)
    temperatura_train =vTemperatura.loc[y_train.index]
    temperatura_test =vTemperatura.loc[y_truth.index]
    vTemperaturaMin.index=pd.to_datetime(vTemperaturaMin.index)
    vTemperaturaMin=vTemperaturaMin.resample('D').sum()
    temperaturaMin_train =vTemperaturaMin.loc[y_train.index]
    temperaturaMin_test =vTemperaturaMin.loc[y_truth.index]
    vTemperaturaMax.index=pd.to_datetime(vTemperaturaMax.index)
    vTemperaturaMax=vTemperaturaMax.resample('D').sum()
    temperaturaMax_train =vTemperaturaMax.loc[y_train.index]
    temperaturaMax_test =vTemperaturaMax.loc[y_truth.index]
    covid_train=vCovid.loc[y_train.index]
    covid_test=vCovid.loc[y_truth.index] 
    vLabor.index=pd.to_datetime(vLabor.index)
    lab_train=vLabor.loc[y_train.index]
    lab_test=vLabor.loc[y_truth.index]
    mejores_parametros= order#[order.order, order.seasonal_order] #ARIMA(1,0,1)(0,1,1)[7]
    temp_train=(temperatura_train+temperaturaMin_train+temperaturaMax_train)/3
    temp_test=(temperatura_test+temperaturaMin_test+temperaturaMax_test)/3

    mod = sm.tsa.statespace.SARIMAX(
                        y_train, 
                        exog=pd.DataFrame({'temperatura': temperatura_train, 'laboralidad':covid_train}),
                        order=mejores_parametros[0],
                        seasonal_order=mejores_parametros[1],
                        enforce_invertibility=False,
                        )
    results = mod.fit()
    pred_uc = results.get_forecast(steps=diasPrediccion, exog=pd.DataFrame({'temperatura': temperatura_test, 'laboralidad':covid_test}))
    pred_ci = pred_uc.conf_int()
    ax = y[y_truth.index].plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(label='Forecast')
    ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Ratio')
    plt.legend()
    plt.title(' TFA:'+str(tarifa)+' CCC:'+ccc)
    plt.show()
    # Evaluación de la predicción ARIMA
    print('ARIMA con regresores para TFA '+str(tarifa)+ ' en comercializadora '+ ccc)
    print(order)
    predicciones_arima = pred_uc.predicted_mean[y_truth.index]
    mse = ((predicciones_arima - y_truth) ** 2).mean()
    rele = (np.abs(predicciones_arima - y_truth)/y_truth*100).mean()
    print('Error cuadrático medio ARIMA {}'.format(round(mse, 2)))
    print('Raíz cuadrada de ECM ARIMA {}'.format(round(np.sqrt(mse), 1))) #reducido un grado
    print('Error porcentual medio ARIMA {}'.format(round(np.sqrt(rele), 1))) #reducido un grado
    return round(np.sqrt(rele), 1)

#Función para realizar predicciones y evaluar ARIMA sin exógenas en distintos escenarios
def prediccionRatioARIMA_sinExog(y_train, y_truth, diasPrediccion, tarifa,ccc, order):
    mejores_parametros= order
    mod = sm.tsa.statespace.SARIMAX(
                                    y_train,
                                    order=mejores_parametros[0],
                                    seasonal_order=mejores_parametros[1],
                                    enforce_invertibility=False,
                                    )
    results = mod.fit()
    residuals = results.resid
    pred_uc = results.get_forecast(steps=diasPrediccion)
    pred_ci = pred_uc.conf_int()
    ax = y[y_truth.index].plot(label='observed', figsize=(14, 7))
    
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Ratio')
    plt.legend()
    plt.title(' sin exog TFA:'+str(tarifa)+' ccc:'+ccc)
    plt.show()
    print('ARIMA sin regresores para '+str(tarifa)+ ' en comercializadora '+ ccc)
    print(order)
    predicciones_arima = pred_uc.predicted_mean[y_truth.index]
    mse = ((predicciones_arima - y_truth) ** 2).mean()
    rele = (np.abs(predicciones_arima - y_truth)/y_truth*100).mean()
    print('Error cuadrático medio ARIMA {}'.format(round(mse, 2)))
    print('Raíz cuadrada de ECM ARIMA {}'.format(round(np.sqrt(mse), 2))) 
    print('Error porcentual medio ARIMA {}'.format(round(np.sqrt(rele), 2))) 
    return residuals

#Función para determinar los mejores parámetros de ARIMA
def pmARIMA_sinExog(y_train):
    import pmdarima as pm
    # SARIMAX Model
    sxmodel = pm.auto_arima(y_train, 
                            #exog=[vLaboralidad[y_train.index], vTemperatura['2021-08-10':'2022-08-24']],
                            start_p=1, start_q=1,
                            test='adf',
                            max_p=10, max_q=10, m=7,
                            start_P=0, seasonal=True,
                            d=None, D=1, trace=True, #True
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
    return sxmodel


import matplotlib
import statsmodels.api as sm
import pandas as pd
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

mejores_parametros={2:[(1,0,1),(2,1,0,7)], 3:[(1,0,1),(0,1,1,7)]}
df_3=pd.read_csv('ResB2C-historico.csv', header=0, index_col=0, sep=';',  decimal='.', encoding='UTF-8', dtype={'FechaLectura': 'str', 'Provincia': 'str', 'ccc': 'str', 'TFA': 'int', 'Wh': 'int', 'IdPS': 'int'})
df_agg=df_3.groupby(["FechaLectura","ccc","TFA"], as_index=False).aggregate({'Wh':'sum', 'IdPS':'sum'})
df_agg['diaLectura']=pd.to_datetime(df_3['FechaLectura']).apply(lambda x: x.date())
df_agg['Ratio']=df_agg['Wh']/df_agg['IdPS']



for ccc in ['CCC']:#,'' ,'']:        para cada comercializadora
    df_agg_ccc=df_agg[df_agg['ccc']==ccc]
    for tarifa in [2,3]:                    #para las tarifas domésticas y comerciales
        #Reagrupamos, agregando horariamente
        df_agg_tfa=df_agg_ccc[df_agg_ccc['TFA']==tarifa]
        df_agg_tfa=df_agg_tfa.set_index(pd.to_datetime(df_agg_tfa['FechaLectura']))
        df_agg_tfa=df_agg_tfa.resample('H').aggregate({'Wh':'sum', 'IdPS':'sum'})
        #Reemplazamos los valores perdidos
        df_agg_tfa['Wh']=replaceValues(df_agg_tfa['Wh'])
        df_agg_tfa['IdPS']=replaceValues(df_agg_tfa['IdPS'])
        df_agg_tfa['Ratio']=np.round(df_agg_tfa['Wh']/df_agg_tfa['IdPS'],2)
        dfRatio=df_agg_tfa.copy()
        #Reagrupamos, agregando diariamiente
        dfRatio=dfRatio.resample('D').aggregate({'Wh':'sum', 'IdPS':'sum'})
        dfRatio['Ratio']=np.round(dfRatio['Wh']/dfRatio['IdPS'],2)
        #Utilizamos la variable de ratio como variable objetivo 'y'
        y =dfRatio['Ratio']
        from pylab import rcParams
        rcParams['figure.figsize'] = 18, 8
        t=y.copy()
        decomposition = sm.tsa.seasonal_decompose(t, model='additive')
        y.plot()
        plt.show()
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(y, lags=14)
        plt.show()
        #Generamos las series de variables exógenas
        dias=pd.Series(index=y.index, data=y.index.values)
        vCovid=dias.copy().map(lambda x: covid(x.date()))
        vLaboralidad=dias.copy().map(lambda x: diaLaborable(x.date()))
        import pmdarima as pm
        #Para cada escenario, creamos un conjunto de datos de entrenamiento y un conjunto de datos de validación para evaluar las predicciones
        for fechas in [['2022-08-10','2022-08-11','2022-08-24'],['2022-10-10','2022-10-11','2022-10-24'],['2022-12-15','2022-12-16','2022-12-27'],['2023-01-31','2023-02-01','2023-02-11']]:
            comienzo={2:'2021-02-10', 3:'2021-09-01'}
            y_train = y.loc[:fechas[0]]
            y_truth = y.loc[fechas[1]:fechas[2]]
            diasPrediccion=len(y_truth)
            order=mejores_parametros[tarifa] #Cargamos aquellos parámetros óptimos calculados por autoarima
            prepararTemperatura() #Se prepara la temperatura
            temperatura_umbral = np.mean(vTemperaturas[ccc])
            temperatura_cuadratica = (vTemperaturas[ccc] -temperatura_umbral) ** 2 #Se transforma la temperatura
            #prediccionRatioARIMA(y_train, y_truth, diasPrediccion, vLaboralidad, vTemperaturas[ccc], vTemperaturaMin[ccc], vTemperaturaMax[ccc], vCovid, tarifa,ccc, order)
            prediccionRatioARIMA_tempCuadratica(y_train, y_truth, diasPrediccion, vLaboralidad, temperatura_cuadratica, vCovid, tarifa, ccc, order)