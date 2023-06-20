import matplotlib
import statsmodels.api as sm
import pandas as pd
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings("ignore")

#Cargar dataset
dfHistorico=pd.read_csv('ResB2C-historico.csv', header=0, sep=';', decimal='.',encoding='utf-8', index_col=0, dtype={'ccc': 'str', 'diaLectura': 'str', 'TFA': 'int', 'Wh': 'float', 'IdPS': 'int', 'Ratio':'float'})
dfHistorico['FechaLectura']=pd.to_datetime(dfHistorico['FechaLectura'], format='%Y-%m-%d %H:%M:%S')

#Función para rellenar con la media nacional en caso de que haya alguna provincia sin medida
def fillna_with_spain(row):
    if pd.isnull(row['España']):
        return row
    else:
        row.fillna(value=row['España'], inplace=True)
        return row

#Función para utilizar las temperaturas de forma ponderada, en función de la distribución de la cartera de clientes
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
    dfTempsMedia = dfTempsMedia.apply(fillna_with_spain, axis=1)
    vTemperaturas['CCC'] = np.round((dfTempsMedia[dfTempsMedia.columns.intersection(diccionarioPonderacionCCC.keys())] * pd.Series(diccionarioPonderacionCCC)).sum(axis=1),2)
    
#Función para distinguir los días laborables/no laborables
def diaLaborable(dia):
    if(dia.isoweekday()>5):
        lab=0
    else:
        lab=1 #Sábados y domingos
    if((dia.day,dia.month) in [(1,1),(6,1),(15,8),(12,10),(1,11),(6,12),(8,12)]):
        lab=1
    return lab

#Función para distinguir los días en los que había una entrada de cliente grande en TARIFA 3
def clienteGrande(dia):
    cg=0
    if((dia.year,dia.month) in [(2021,6),(2021,7)]):
        cg=1
    if((dia.year,dia.month) in [(2021,8)] and dia.day<22):
        cg=1
    else:
        cg=0
    return cg

def diaReferencia(dia, semanas=1):
    try:
        return dfRatio['Ratio'][dia - np.timedelta64(7*semanas,'D')]
    except:
        return None

def covid(dia):
    if(dia<datetime.strptime('2021-02-15',"%Y-%m-%d").date()):
        return 1
    if(dia >datetime.strptime('2021-12-15',"%Y-%m-%d").date() and dia<datetime.strptime('2022-01-15',"%Y-%m-%d").date()):
        return 0
    else:
        return 0

#Reemplazar valores anómalos tales como los derivados de cambios de hora o valores perdidos
def replaceValues(data):
    data_reemplazado = data.replace(0, np.nan)
    for cambiodehora in ['2021-03-28 00:00:00', '2021-10-31 01:00:00', '2022-03-27- 01:00:00']:
        data_reemplazado.loc[cambiodehora]=None  
    data_reemplazado  = data_reemplazado.interpolate(method='linear')
    for nulos in ['2022-05-26']:
        data_reemplazado.loc[nulos]=data_reemplazado.loc['2022-05-25'].values
    return data_reemplazado

#Modelación de días festivos
def cargarFestivos(ccc, diasCovid):
    # Festivos con impacto grande
    festivos_grandes = [
        {
            'holiday': 'Nochevieja y Año Nuevo',
            'ds': pd.to_datetime(['2021-12-31', '2022-12-31', '2023-12-31']).to_list(),
            'lower_window': 0,
            'upper_window': 1,
        },
        {
            'holiday': 'Nochebuena',
            'ds': pd.to_datetime(['2020-12-24', '2021-12-24', '2022-12-24', '2023-12-24']).to_list(),
            'lower_window': -1,
            'upper_window': 1,
        },   
    ]
    # Festivos con impacto menor
    festivos_menores = [
        {
            'holiday': 'Día del Trabajo',
            'ds': pd.to_datetime(['2021-05-01','2022-05-01','2023-05-01']).to_list(),
            'lower_window': -1,
            'upper_window': 0,
        },
        {
            'holiday': 'Día Nacional',
            'ds': pd.to_datetime(['2021-10-12','2022-10-12','2023-10-12']).to_list(),
            'lower_window': 0,
            'upper_window': 0,
        },
        {
            'holiday': 'Día Constitución e Inmaculada',
            'ds': pd.to_datetime(['2021-12-06', '2021-12-08','2022-12-06', '2022-12-08','2023-12-06', '2023-12-08']).to_list(),
            'lower_window': 0,
            'upper_window': 0,
        },
        
        {
            'holiday': 'Reyes',
            'ds': pd.to_datetime(['2021-12-06', '2022-12-06', '2023-12-06']).to_list(),
            'lower_window': -1,
            'upper_window': 0,
        },
        {
            'holiday': 'Semana santa (Viernes Santo)',
            'ds': pd.to_datetime(['2021-04-02', '2022-04-15', '2023-04-07']).to_list(),
            'lower_window': -1,
            'upper_window': 0,
        },

        {
            'holiday': '15 agosto',
            'ds': pd.to_datetime(['2022-08-15', '2023-08-15']).to_list(),
            'lower_window': 0,
            'upper_window': 0,
        },
        
        {
            'holiday': 'Todos los Santos',
            'ds': pd.to_datetime(['2021-11-01', '2022-11-01', '2023-11-01']).to_list(),
            'lower_window': -1,
            'upper_window': 0,
        }
        ,{
            'holiday': 'Covid',
            'ds': pd.to_datetime(diasCovid).to_list(),
            'lower_window': 0,
            'upper_window': 0,
        }
        
    ]
    # Crear DataFrames para festivos grandes y menores
    df_festivos_grandes = pd.DataFrame(festivos_grandes)
    df_festivos_menores = pd.DataFrame(festivos_menores)
    # Fusionar DataFrames de festivos
    df_festivos = pd.concat([df_festivos_grandes, df_festivos_menores], ignore_index=True)
    festivos_df = pd.DataFrame(
    [
        {
            'holiday': festivo['holiday'],
            'ds': fecha,
            'lower_window': festivo['lower_window'],
            'upper_window': festivo['upper_window'],
        }
        for _, festivo in df_festivos.iterrows()
        for fecha in festivo['ds']
    ]
    )   
    return festivos_df


def prediccionRatioProphet(y_train, y_truth,tarifa,ccc):
    combinado_train=pd.DataFrame()
    combinado_train=y_train
    combinado_train['y']=combinado_train['Ratio']
    combinado_train['ds']=combinado_train.index
    from prophet import Prophet
    diasCovid=y_train[y_train['covid']==1].index
    combinado_model = Prophet(
                #interval_width=0.95
                #growth='linear',
                holidays=cargarFestivos(ccc, diasCovid),
                yearly_seasonality=True,#, changepoint_prior_scale=0.01)
                changepoint_prior_scale=0.5,
                #seasonality_prior_scale=10,
                #holidays_prior_scale=5,
                seasonality_mode='additive')
    #combinado_model.add_country_holidays(country_name='ES')
    #combinado_model.seasonality_mode = 'multiplicative'
    #combinado_model.add_regressor("Laboralidad", standardize=False)
    combinado_model.add_regressor("Temperatura", standardize=True)
    combinado_model.fit(combinado_train) #Entrenar modelo
    y_truth['ds']=y_truth.index
    forecast = combinado_model.predict(y_truth)
    forecast.index=forecast['ds']
    predicciones_prophet2=forecast.loc[y_truth.index, "yhat"]
    combinado_model.plot_components(forecast)
    plt.show()
    # PROPHET errores
    # ------------------------------------
    y_truth['y']=y_truth['Ratio']
    mse = ((predicciones_prophet2 - y_truth['y']) ** 2).mean()
    rele = (np.abs(predicciones_prophet2 - y_truth['y'])/y_truth['y']*100).mean()
    print('Error cuadrático medio PROPHET {}'.format(round(mse, 1)))
    print('Raíz cuadrada de ECM PROPHET {}'.format(round(np.sqrt(mse), 1)))
    print('Error porcentual medio PROPHET {}'.format(round(np.sqrt(rele), 1)))
    plt.plot(y_truth['y'], label="Fact")
    plt.plot(dfRatio['diaReferencia'][y_truth.index], label='Actual system')
    plt.plot(predicciones_prophet2, label="Prophet Forecast")
    plt.legend()
    plt.title('Ratio prediction en TFA'+str(tarifa)+ ' para '+ccc)
    plt.show()
    return round(np.sqrt(rele), 1)

def prediccionRatioNeuralProphet(y_train, y_truth,tarifa,ccc):
    combinado_train=pd.DataFrame()
    combinado_train=y_train
    combinado_train['y']=combinado_train['Ratio']
    combinado_train['ds']=combinado_train.index
    y_truth['y']=y_truth['Ratio']
    y_truth['ds']=y_truth.index

    from neuralprophet import NeuralProphet
    combinado_model = NeuralProphet(
        growth="linear", 
        #changepoints=None, 
        #n_changepoints=200,
        changepoints_range=0.9,
        #trend_reg=0.8,
        #trend_reg_threshold=False,
        #yearly_seasonality=True,
        #weekly_seasonality=True,
        #daily_seasonality="auto",
        #seasonality_mode="additive",
        #seasonality_reg=0,
        #n_forecasts=1,
        #n_lags=p,
        #batch_size=3,
        #atch_size=64,
        #learning_rate=0.01,
        loss_func="Huber",
        #normalize="standardize",  # Type of normalization ('minmax', 'standardize', 'soft', 'off')
        impute_missing=True
        
    )
    diasCovid=y_train[y_train['covid']==1].index
    combinado_train = combinado_train.drop(columns=[col for col in combinado_train.columns if col not in ['Temperatura', 'ds', 'y']])
    y_truth = y_truth.drop(columns=[col for col in y_truth.columns if col not in ['Temperatura','ds', 'y']])
    combinado_train=combinado_train.dropna(subset=['y'])
    
    df_events = pd.DataFrame({
    'event': ['covid','Nochevieja y Año Nuevo', 'Nochebuena', 'Día del Trabajo', 'Día Nacional', 'Día Constitución e Inmaculada','Reyes','Semana santa (Viernes Santo)', 'Todos los Santos', '15Agosto', 'regionales'],
    'ds': [
        pd.to_datetime(diasCovid.values).to_list(),
        pd.to_datetime(['2021-12-31', '2022-12-31', '2023-12-31']).to_list(),
        pd.to_datetime(['2020-12-24', '2021-12-24', '2022-12-24', '2023-12-24']).to_list(),
        pd.to_datetime(['2021-05-01','2022-05-01','2023-05-01']).to_list(),
        pd.to_datetime(['2021-10-12','2022-10-12','2023-10-12']).to_list(),
        pd.to_datetime(['2021-12-06', '2021-12-08','2022-12-06', '2022-12-08','2023-12-06', '2023-12-08']).to_list(),
        pd.to_datetime(['2021-12-06', '2022-12-06', '2023-12-06']).to_list(),
        pd.to_datetime(['2021-04-02', '2022-04-15', '2023-04-07']).to_list(),
        pd.to_datetime(['2021-11-01', '2022-11-01', '2023-11-01']).to_list(),
        pd.to_datetime(['2022-08-15', '2023-08-15']).to_list(),
        #pd.to_datetime(dicRegionales[ccc]).to_list()
    ]
    })
         
    #combinado_model = combinado_model.add_country_holidays(country_name='ES')
    combinado_model.add_events(['covid','Nochevieja y Año Nuevo', 'Nochebuena', 'Día del Trabajo', 'Día Nacional', 'Día Constitución e Inmaculada','Reyes','Semana santa (Viernes Santo)', 'Todos los Santos','15Agosto','regionales'],lower_window=-1,upper_window=1)
    combinado_train_all = combinado_model.create_df_with_events(combinado_train, df_events)
    combinado_model= combinado_model.add_lagged_regressor("Temperatura", normalize="standardize")
    print(combinado_train)
    combinado_model.set_plotting_backend("plotly")
    #combinado_model = combinado_model.add_country_holidays("ES", lower_window=-1, upper_window=0)
    metrics=combinado_model.fit(combinado_train_all,  freq="D")#, progress="plot") #Entrenar modelo
    print(combinado_model.config_train)
    y_truth['ds']=y_truth.index
    forecast = combinado_model.predict(y_truth)
    a=combinado_model.plot_components(forecast)
    import plotly.offline as pyo
    pyo.plot(a)
    plt.show()

    forecast.index=forecast['ds']
    predicciones_prophet2=forecast.loc[y_truth.index, "yhat1"]
    # PROPHET múltiple
    # ------------------------------------
    mse = ((predicciones_prophet2 - y_truth['y']) ** 2).mean()
    rele = (np.abs(predicciones_prophet2 - y_truth['y'])/y_truth['y']*100).mean()
    print('Error cuadrático medio PROPHET múltiple {}'.format(round(mse, 1)))
    print('Raíz cuadrada de ECM PROPHET múltiple {}'.format(round(np.sqrt(mse), 1)))
    print('Error porcentual medio PROPHET múltiple {}'.format(round(np.sqrt(rele), 1)))
    plt.plot(y_truth['y'], label="Fact")
    #plt.plot(dfRatio['diaReferencia'][y_truth.index], label='Actual system')
    plt.plot(predicciones_prophet2, label="Prophet Forecast")
    #plt.plot(actualSystem)
    #plt.ylim(0)
    plt.legend()
    plt.title('Ratio prediction en TFA'+str(tarifa)+ ' para '+ccc)
    plt.show()
    return round(np.sqrt(rele), 1)

df_3=pd.read_csv('ResB2C-historico.csv', header=0, index_col=0, sep=';',  decimal='.', encoding='UTF-8', dtype={'FechaLectura': 'str', 'Provincia': 'str', 'ccc': 'str', 'TFA': 'int', 'Wh': 'int', 'IdPS': 'int'})
df_agg=df_3.groupby(["FechaLectura","ccc","TFA"], as_index=False).aggregate({'Wh':'sum', 'IdPS':'sum'})
df_agg['diaLectura']=pd.to_datetime(df_3['FechaLectura']).apply(lambda x: x.date())
df_agg['Ratio']=df_agg['Wh']/df_agg['IdPS']

errores=dict()
for ccc in ['CCC', '', '']: #
    df_agg_ccc=df_agg[df_agg['ccc']==ccc]
    for fechas in [['2022-08-10','2022-08-11','2022-08-26'],['2022-10-10','2022-10-11','2022-10-26'],['2022-12-15','2022-12-16','2023-01-01'],['2023-01-31','2023-02-01','2023-02-11']]:
        for tarifa in [2,3]:#
            #Agrupación en serie de medidas diaria
            df_agg_tfa=df_agg_ccc[df_agg_ccc['TFA']==tarifa]
            df_agg_tfa=df_agg_tfa.set_index(pd.to_datetime(df_agg_tfa['FechaLectura']))
            df_agg_tfa=df_agg_tfa.resample('H').aggregate({'Wh':'sum', 'IdPS':'sum'})
            df_agg_tfa['Wh']=replaceValues(df_agg_tfa['Wh'])
            df_agg_tfa['IdPS']=replaceValues(df_agg_tfa['IdPS'])
            df_agg_tfa['Ratio']=np.round(df_agg_tfa['Wh']/df_agg_tfa['IdPS'],2)
            dfRatio=df_agg_tfa.copy()        
            dfRatio=dfRatio.resample('D').aggregate({'Wh':'sum', 'IdPS':'sum'})
            dfRatio['diaLectura']=dfRatio.index.values
            dfRatio['diaLectura']=dfRatio['diaLectura'].map(lambda x: x.date())
            dfRatio['Ratio']=np.round(dfRatio['Wh']/dfRatio['IdPS'],2)
            #Prueba de valores a utilizar como variables exógenas
            dfRatio['diaReferencia']=dfRatio.apply(lambda row: diaReferencia(row['diaLectura'],1), axis=1)
            dfRatio['covid']=dfRatio.apply(lambda row: covid(row['diaLectura']), axis=1)
            dfRatio['Laboralidad']=dfRatio.apply(lambda row: diaLaborable(row['diaLectura']), axis=1)
            dfRatio['clienteGrande']=dfRatio.apply(lambda row: clienteGrande(row['diaLectura']), axis=1)
            prepararTemperatura()
            temperatura_umbral = np.mean(vTemperaturas[ccc].loc[dfRatio.index])
            temperatura_cuadratica = (vTemperaturas[ccc] -temperatura_umbral) ** 2 
            dfRatio['Temperatura']= temperatura_cuadratica
            comienzo={2:'2021-02-10', 3:'2021-09-01'}
            #Distinción de conjuntos de entrenamiento y validación
            y_train = dfRatio.loc[:fechas[0]].copy()
            from datetime import datetime, timedelta
            horizonteprev=len(dfRatio.loc[fechas[1]:fechas[2]])
            p=horizonteprev
            p=1
            inicio=datetime.strptime(fechas[1], '%Y-%m-%d')- timedelta(days=p) ##prophet
            y_truth = dfRatio.loc[inicio:fechas[2]].copy()
            #predicción de temperatura
            laborables=dfRatio.loc[y_train.index]
            laborables=laborables[laborables['Laboralidad']==0]
            laborablesSinCovid=laborables[laborables['covid']==0]
            if(tarifa==3):
                laborablesSinCovid=laborablesSinCovid[laborablesSinCovid['clienteGrande']==0]
            coeficientes = np.polyfit(vTemperaturas[ccc].loc[laborablesSinCovid.index], dfRatio.loc[laborablesSinCovid.index]['Ratio'], 2)
            polinomio = np.poly1d(coeficientes)
            prediccionTemperaturas=vTemperaturas[ccc].loc[dfRatio.index].apply(lambda temp: polinomio(temp))
            dfRatio['Temperatura']= prediccionTemperaturas
            y_truth = dfRatio.loc[inicio:fechas[2]].copy()
            print(y_truth)
            prediccionRatioProphet(y_train, y_truth, tarifa, ccc)
            prediccionRatioNeuralProphet(y_train, y_truth,tarifa,ccc)
