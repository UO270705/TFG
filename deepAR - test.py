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
dfHistorico_provincias=pd.read_csv('dataset_agregado_temps06.csv', header=0, sep=';', decimal='.',encoding='utf-8', index_col=0, dtype={'ccc': 'str', 'FechaLectura': 'str', 'Provincia': 'str', 'TFA': 'int', 'Wh': 'float', 'IdPS': 'int', 'Ratio':'float', 'TipoDia':'int', 'Hora':'int','Estacion':'str','Temperatura':'float'})
dfHistorico_provincias['FechaLectura']=pd.to_datetime(dfHistorico_provincias['FechaLectura'], format='%Y-%m-%d %H:%M:%S')
dfHistorico['FechaLectura']=pd.to_datetime(dfHistorico['FechaLectura'], format='%Y-%m-%d %H:%M:%S')
dfHistorico_provincias=dfHistorico_provincias[dfHistorico_provincias['TFA']==2]

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
    # Aplicar la función a cada fila del DataFrame
    dfTempsMedia = dfTempsMedia.apply(fillna_with_spain, axis=1)
    vTemperaturas['CCC'] = np.round((dfTempsMedia[dfTempsMedia.columns.intersection(diccionarioPonderacionCCC.keys())] * pd.Series(diccionarioPonderacionCCC)).sum(axis=1),2)

#Función para distinguir los días laborables/no laborables
def diaLaborable(dia):
    if(dia.isoweekday()>5):#Sábados y domingos
        lab=0
    else:
        lab=1 
    if((dia.day,dia.month) in [(1,1),(6,1),(15,8),(12,10),(1,11),(6,12),(8,12)]):
        lab=0
    return lab

def diaReferencia(dia, semanas=1):
    try:
        return dfRatio['Ratio'][dia - np.timedelta64(7*semanas,'D')]
    except:
        return None

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

def prediccionDeepAR(y_train, y_truth,y_truth_len,tarifa,ccc,incluirprovincias):
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.util import to_pandas
    from gluonts.torch.model.deepar import DeepAREstimator
    from gluonts.evaluation.backtest import make_evaluation_predictions
    import torch
    dfHistorico_provincias=pd.read_csv('dataset_agregado_temps06.csv', header=0, sep=';', decimal='.',encoding='utf-8', index_col=0, dtype={'ccc': 'str', 'FechaLectura': 'str', 'Provincia': 'str', 'TFA': 'int', 'Wh': 'float', 'IdPS': 'int', 'Ratio':'float', 'TipoDia':'int', 'Hora':'int','Estacion':'str','Temperatura':'float'})
    dfHistorico_provincias['FechaLectura']=pd.to_datetime(dfHistorico_provincias['FechaLectura'], format='%Y-%m-%d %H:%M:%S')
    comienzo={2:'2021-02-10', 3:'2021-09-01'}
    dfHistorico_provincias=dfHistorico_provincias[dfHistorico_provincias['TFA']==2]
    dfHistorico_provincias['FechaLectura'] =  pd.to_datetime(dfHistorico_provincias['FechaLectura'], format='%Y-%m-%d %H:%M:%S.%f') #Tratar como fechas
    dfHistorico_provincias['diaLectura'] =dfHistorico_provincias['FechaLectura'].dt.date 
    dfHistorico_provincias=dfHistorico_provincias[dfHistorico_provincias['TFA']==2]
    dfHistorico_provincias=dfHistorico_provincias.groupby(['ccc','Provincia', 'diaLectura'], as_index=False).aggregate({'Wh':'sum', 'IdPS':'sum', 'Temperatura':'mean'})
    dfHistorico_provincias['diaLectura'] =  pd.to_datetime(dfHistorico_provincias['diaLectura'], format='%Y-%m-%d') #Tratar como fechas
    dfHistorico_provincias['Ratio']=dfHistorico_provincias['Wh']/dfHistorico_provincias['IdPS']
    print("----------------")
    print(df_agg)
    series=[]
    c=0
    if(incluirprovincias):
        for ccc in ['CCC']:
            df_aggCCC=dfHistorico_provincias[dfHistorico_provincias['ccc']==ccc]
            for provincia in dfHistorico_provincias['Provincia'].unique():
                serie=df_aggCCC[df_aggCCC['Provincia']==provincia].set_index('diaLectura').resample('D').sum().bfill()
                serie=serie.loc[comienzo[tarifa]:]
                if(serie['IdPS'].mean()>=30000): #Si la serie tiene de media más de 30000 IdPS
                    laborables=serie
                    serie['dia']=serie.index
                    serie['Laboralidad']=serie.apply(lambda row: diaLaborable(row['dia']), axis=1)
                    serie['covid']=serie.apply(lambda row: covid(row['dia']), axis=1)
                    laborables=laborables[laborables['Laboralidad']==0]
                    laborablesSinCovid=laborables[laborables['covid']==0]
                    if(len(laborablesSinCovid)>10):
                        temperatura_umbral = np.mean(serie['Temperatura'])
                        temperatura_cuadratica = (serie['Temperatura'] -temperatura_umbral) ** 2
                        serie['Temperatura']=temperatura_cuadratica
                        dataset_serie={
                            "start": str(serie.index[0]),
                            "target": serie['Ratio'].values,
                            "feat_dynamic_real": [serie['Temperatura'].values],
                            "feat_static_cat":[0]
                            }
                        series.append(dataset_serie)
    #series=[]    
    series.append({"start": str(y_train.index[0]), "target": y_train['Ratio'].values, "feat_dynamic_real": [y_train['Temperatura'].values], 'feat_static_cat':[0]}) #'feat_static_cat': y_train['covid'].values,)
    training_data = ListDataset([serie for serie in series],freq="D")
    
    validation_data = ListDataset([{#feat_static_real
        "start": str(y_truth.index[0]), "target": y_truth["Ratio"].values, "feat_dynamic_real": [y_truth['Temperatura'].values], "feat_static_cat":[0],#  "feat_static_cat": y_truth['covid'].values,
        }],
        freq="D")
    
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=y_truth_len,
        context_length=y_truth_len,
        dropout_rate=0.01,
        num_feat_dynamic_real=1,
        num_feat_static_cat=1,
        cardinality=[50],
        num_layers=3,
        trainer_kwargs={"max_epochs": 50},
        batch_size=32
    )
    predictor = estimator.train(training_data=training_data)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=validation_data,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)
    ts_entry = tss[0]
    forecast_entry = forecasts[0]   
    def plot_prob_forecasts(ts_entry, forecast_entry):
        plot_length = 150
        prediction_intervals = (50.0, 90.0)
        legend = ["observations", "median prediction"] + \
            [f"{k}% prediction interval" for k in prediction_intervals][::-1]

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ts_entry.plot(ax=ax) 
        forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
        plt.grid(which="both")
        plt.legend(legend, loc="upper left")
        plt.show()
        #nombre_archivo='variasSeries'+str(y_truth.index[0].strftime("%Y-%m-%d"))+'_batch'+str(b)+'_epoch '+str(e)+'__'+str(n)
        ax.set_title('Ratio prediction en TFA'+str(tarifa)+ ' para '+ccc)
        #ruta_guardado = f"figuras/{nombre_archivo}"
        #plt.savefig(ruta_guardado)
        # Cerrar la figura
        #plt.close(fig)

    
    plot_prob_forecasts(ts_entry, forecast_entry)
    
    y_truth=y_truth[y_train.index[-1]:][1:]
    y_truth['y']=y_truth['Ratio']
    predicciones_deepar = forecast_entry.mean_ts
    mse = ((predicciones_deepar.values - y_truth['y'].values) ** 2).mean()
    rele = (np.abs(predicciones_deepar.values - y_truth['y'].values)/y_truth['y'].values*100).mean()
    print("Resultados para el escenario de "+str(y_truth.index[0]))
    print('Error cuadrático medio DEEPAR {}'.format(round(mse, 2)))
    print('Raíz cuadrada de ECM DEEPAR {}'.format(round(np.sqrt(mse), 2)))
    print('Error porcentual medio DEEPAR {}'.format(round(np.sqrt(rele), 2)))
    

    plt.plot(y_truth['y'], label="Fact")
    plt.plot(predicciones_deepar, label="DeepAR Forecast")
    plt.legend()
    plt.title('Ratio prediction en TFA'+str(tarifa)+ ' para '+ccc)
    plt.show()
    return round(np.sqrt(rele), 1)


df_3=pd.read_csv('ResB2C-historico.csv', header=0, index_col=0, sep=';',  decimal='.', encoding='UTF-8', dtype={'FechaLectura': 'str', 'Provincia': 'str', 'ccc': 'str', 'TFA': 'int', 'Wh': 'int', 'IdPS': 'int'})
df_agg=df_3.groupby(["FechaLectura","ccc","TFA"], as_index=False).aggregate({'Wh':'sum', 'IdPS':'sum'})
df_agg['diaLectura']=pd.to_datetime(df_3['FechaLectura']).apply(lambda x: x.date())
df_agg['Ratio']=df_agg['Wh']/df_agg['IdPS']

errores=dict()
for ccc in ['CCC']:#, '', '']: #
    #df_agg=df_3
    df_agg_ccc=df_agg[df_agg['ccc']==ccc]
    for fechas in [['2022-08-10','2022-08-11','2022-08-26'],['2022-10-10','2022-10-11','2022-10-26'],['2022-12-15','2022-12-16','2023-01-01'],['2023-01-31','2023-02-01','2023-02-11']]:
        for tarifa in [2]:#,3]: #
            df_agg_tfa=df_agg_ccc[df_agg_ccc['TFA']==tarifa]
            df_agg_tfa=df_agg_tfa.set_index(pd.to_datetime(df_agg_tfa['FechaLectura']))
            df_agg_tfa=df_agg_tfa.resample('H').aggregate({'Wh':'sum', 'IdPS':'sum'})
            df_agg_tfa['Wh']=replaceValues(df_agg_tfa['Wh'])
            df_agg_tfa['IdPS']=replaceValues(df_agg_tfa['IdPS'])
            df_agg_tfa['Ratio']=np.round(df_agg_tfa['Wh']/df_agg_tfa['IdPS'],2)
            dfRatio=df_agg_tfa.copy()
            dfRatio=dfRatio.resample('D').aggregate({'Wh':'sum', 'IdPS':'sum'})
            #Seleccion de variables exógenas
            dfRatio['diaLectura']=dfRatio.index.values
            dfRatio['diaLectura']=dfRatio['diaLectura'].map(lambda x: x.date())
            dfRatio['Ratio']=np.round(dfRatio['Wh']/dfRatio['IdPS'],2)
            dfRatio['diaReferencia']=dfRatio.apply(lambda row: diaReferencia(row['diaLectura'],1), axis=1)
            dfRatio['covid']=dfRatio.apply(lambda row: covid(row['diaLectura']), axis=1)
            dfRatio['Laboralidad']=dfRatio.apply(lambda row: diaLaborable(row['diaLectura']), axis=1)
            dfRatio['y_lag'] = dfRatio['Ratio'].shift(14)
            dfRatio['y_lag']=dfRatio['y_lag'].fillna(0)
            prepararTemperatura()
            temperatura_umbral = np.mean(vTemperaturas[ccc].loc[dfRatio.index])
            temperatura_cuadratica = (vTemperaturas[ccc] -temperatura_umbral) ** 2 
            dfRatio['Temperatura']= vTemperaturas[ccc].loc[dfRatio.index]#temperatura_cuadratica
            comienzo={2:'2021-02-10', 3:'2021-09-01'}
            y_train = dfRatio.loc[:fechas[0]].copy()
            y_truth = dfRatio.loc[fechas[0]:fechas[2]].copy()
            y_truth_len=len(y_truth[fechas[1]:])
            prediccionDeepAR(y_train, y_truth,y_truth_len,tarifa,ccc, True)
