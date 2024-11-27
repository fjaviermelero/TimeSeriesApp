import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

print('Importando datos')
#Importar datos
start_time = time.time()

dfA = pd.read_excel('../data/ERP_data/01.21-12.22.xlsx')
dfB = pd.read_excel('../data/ERP_data/12.22-12.23.xlsx')
dfC = pd.read_excel('../data/ERP_data/12.23-07.24.xlsx')
dfD = pd.read_excel('../data/ERP_data/07.24-11.24.xlsx')

end_time = time.time()

elapsed_time = end_time - start_time
print('Tiempo de carga: ' + str(elapsed_time))

## 1: Limpieza y preparación del Dataset
#===============================================================================================
#Concatenar
df = pd.concat([dfA, dfB, dfC, dfD])

#Eliminar duplicados
df = df.drop_duplicates()

#Renombrar columnas
columnas = ['nodos','fecha_arch', 'hora_arch', 'scanner', 'codigo', 'unidad_manipulac', 'entrega', 'posicion', 'doc_ventas', 'posicion2', 'material', 'denominacion', 'fecha_hist', 'hora_hist']
df = df.set_axis(columnas, axis = 1)

#Eliminar espacios antes y despues de strings
columns_to_strip = ['material', 'denominacion']
for col in columns_to_strip:
    df[col] = df[col].astype(str).str.strip()

#Combinar fechas históricas y archivadas
df = df.reset_index(drop=True)
df['fecha_arch'] = pd.to_datetime(df['fecha_arch'].str.strip(), format='%d.%m.%Y', errors='coerce')
df['fecha_hist'] = pd.to_datetime(df['fecha_hist'].str.strip(), format='%d.%m.%Y', errors='coerce')
df['fecha'] = df['fecha_arch'].combine_first(df['fecha_hist'])

#Combianr horas históricas y archivadas
df = df.reset_index(drop=True)
df['hora_arch'] = df['hora_arch'].astype(str)
df['hora_hist'] = df['hora_hist'].astype(str)
df['hora_arch'] = pd.to_timedelta(df['hora_arch'].str.strip(), errors='coerce')
df['hora_hist'] = pd.to_timedelta(df['hora_hist'].str.strip(), errors='coerce')
df['hora'] = df['hora_arch'].combine_first(df['hora_hist'])

#Eliminar fechas y horas previas a la combinación
df = df.drop(['fecha_arch', 'hora_arch', 'fecha_hist', 'hora_hist'], axis = 1)


# Calcular el turno donde se ha escaneado el producto
def calcular_turno(row):
    if row['hora'] >= pd.to_timedelta('01:00:00') and row['hora'] <= pd.to_timedelta('15:15:00'):
        return 1
    else:
        return 2

df['turno'] = df.apply(calcular_turno, axis=1)

#Calcular mes y año
df['mes'] = df['fecha'].dt.month
df['year'] = df['fecha'].dt.year

lineas = {
"SAPRFMADR26": "Mesas",
"SAPRFMADR20": "Paneles",
"SAPRFMADR23": "Tableros",
"SAPRFMADR16": "Volumenes",
"SAPRFMADR19": "Conectores",
"SAPRFMADR21": "Varios",
"SAPRFMADR17": "Silla DO",   # ARREGLAR DATOS NaN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"SAPRFMADR18": "Pie Ajustable",
"SAPRFMADR15": "Bloques",
"SAPRFMADR24": "Carpinteria otros",
"SAPRFMADR25": "Partito Rail"
}

## 2: Calculo de Dataset para Pronóstico
#============================================================================================================
numeros_scanners = list(lineas.keys())

bandera = False

for numero_scanner_linea in numeros_scanners:

    # Filtrar la linea de interes

    if numero_scanner_linea == 'SAPRFMADR16':

        # Juntamos Volúmenes con bloques
        df_linea = df[(df['scanner'] == 'SAPRFMADR16') | (df['scanner'] == 'SAPRFMADR15')]

    elif numero_scanner_linea == 'SAPRFMADR15':
        continue

    else:
        df_linea = df[(df['scanner'] == numero_scanner_linea)]

    # Calcular diferencia de tiempos entre piezas
    df_linea['delta'] = df_linea['hora'].diff().dt.total_seconds()
    # Agregamos estado que represente si la pieza se ha producido tras un cambio de producto.
    df_linea['cambio'] = (df_linea['material'] != df_linea['material'].shift()).astype(int)
    # Añadimos un estado que represente pausas de > 10 mins
    df_linea['pausa'] = ((df_linea['delta'] > 600) | (df_linea['delta'] < 0)).astype(int)

    # Estudiamos Takt Times de los volumenes filtrando los que se han hecho tras una pausa y tras cambio de producto.
    # Estudiamos Takt Times en turno 1 ya que la mano de obra es aproximádamente constante

    df_linea_takt_filtr = df_linea[(df_linea['cambio'] == 0) & (df_linea['pausa'] == 0) & (df_linea['turno'] == 1)]

    df_linea_takt_filtr_agrup = df_linea_takt_filtr.groupby(
        ['denominacion']
    ).agg({'delta': ['mean', 'std', 'count']}).reset_index()

    df_linea_takt_filtr_agrup.columns = ['denominacion', 'media_takt', 'dev_std', 'fabricados']
    df_linea_takt_filtr_agrup = df_linea_takt_filtr_agrup.sort_values(by='fabricados', ascending=False)

    # Tabla con productos mas fabricados, desviación estandar y Takt Time
    df_linea_takt_filtr_agrup

    # Agrupar por fecha y denominación, y calcular la cantidad fabricada
    df_linea_productos_dia = (
        df_linea.groupby(['fecha', 'denominacion'])['nodos']
            .count()
            .reset_index(name='cantidad_fabricada')
    )

    # Unir con el DataFrame de Takt Time filtrado para añadir el takt time de cada producto
    df_linea_productos_dia = df_linea_productos_dia.merge(
        df_linea_takt_filtr_agrup[['denominacion', 'media_takt']],
        on='denominacion'
    )

    # Calcular el tiempo de trabajo para cada producto
    df_linea_productos_dia['tiempo_trabajo'] = (
            df_linea_productos_dia['cantidad_fabricada'] * df_linea_productos_dia['media_takt']
    )

    # Trabajo Diario
    df_linea_trabajo_diario = df_linea_productos_dia.groupby(by='fecha').sum().reset_index()[
        ['fecha', 'cantidad_fabricada', 'tiempo_trabajo']]
    df_linea_trabajo_diario['tiempo_trabajo_horas'] = df_linea_trabajo_diario['tiempo_trabajo'] / 3600
    df_linea_trabajo_diario['semana'] = df_linea_trabajo_diario['fecha'].dt.strftime('%Y-%U')
    df_linea_trabajo_diario

    # Trabajo Semanal
    # Estudiar por semanas Y rellenar faltantes
    df_linea_trabajo_semanal = df_linea_trabajo_diario.drop('fecha', axis=1).groupby(by='semana').sum().reset_index()


    def semana_a_fecha(semana_str):
        year, week = map(int, semana_str.split('-'))
        # Crear una fecha base el primer día del año
        first_day_of_year = pd.Timestamp(year=year, month=1, day=1)
        # Calcular la fecha del primer día de la semana específica
        return first_day_of_year + pd.to_timedelta(week * 7 - first_day_of_year.dayofweek, unit='D')


    df_linea_trabajo_semanal['semana_primer_dia'] = df_linea_trabajo_semanal['semana'].apply(semana_a_fecha)

    # Rellenar semanas faltantes
    semana_min = df_linea_trabajo_semanal['semana_primer_dia'].min()
    semana_max = df_linea_trabajo_semanal['semana_primer_dia'].max()
    todas_las_semanas = pd.date_range(semana_min, semana_max, freq='W-MON')
    df_todas_las_semanas = pd.DataFrame({'semana_primer_dia': todas_las_semanas})
    df_completo = pd.merge(df_todas_las_semanas, df_linea_trabajo_semanal, on='semana_primer_dia', how='left')
    df_completo['cantidad_fabricada'].fillna(0, inplace=True)
    df_completo['tiempo_trabajo'].fillna(0, inplace=True)
    df_completo['tiempo_trabajo_horas'].fillna(0, inplace=True)
    df_completo['semana'] = df_completo['semana_primer_dia'].dt.isocalendar().week.astype(str) + "-" + df_completo[
        'semana_primer_dia'].dt.year.astype(str)

    df_linea_trabajo_semanal = df_completo

    if bandera == False:

        # Primera linea

        df_linea_trabajo_semanal_conjunto = df_linea_trabajo_semanal

        df_linea_trabajo_semanal_conjunto = df_linea_trabajo_semanal_conjunto.rename(columns={
            'cantidad_fabricada': 'cantidad_fabricada_' + numero_scanner_linea,
            'tiempo_trabajo_horas': 'tiempo_trabajo_horas_' + numero_scanner_linea
        })

        df_linea_trabajo_semanal_conjunto = df_linea_trabajo_semanal_conjunto.drop('tiempo_trabajo', axis=1)

        bandera = True

    else:

        nombre_columna_cantidad = 'cantidad_fabricada_' + numero_scanner_linea
        nombre_columna_tiempo_trabajo = 'tiempo_trabajo_horas_' + numero_scanner_linea

        df_linea_trabajo_semanal_conjunto[nombre_columna_cantidad] = df_linea_trabajo_semanal['cantidad_fabricada']
        df_linea_trabajo_semanal_conjunto[nombre_columna_tiempo_trabajo] = df_linea_trabajo_semanal['tiempo_trabajo_horas']

df_linea_trabajo_semanal_conjunto['tiempo_trabajo_horas_total'] = df_linea_trabajo_semanal_conjunto.filter(
    like='tiempo_trabajo_horas_').sum(axis=1)
df_linea_trabajo_semanal_conjunto['cantidad_fabricada_total'] = df_linea_trabajo_semanal_conjunto.filter(
    like='cantidad_fabricada_').sum(axis=1)

## 3: Pronóstico con Prophet
#========================================================================================================

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

numeros_scanners.append('total')

precision_Prophet = pd.DataFrame()
predicciones_Prophet = pd.DataFrame()

for numero_scanner_linea in numeros_scanners:

    print(numero_scanner_linea)

    # Preparamos datos.

    if numero_scanner_linea == 'SAPRFMADR15' or numero_scanner_linea == 'SAPRFMADR17':
        continue

    nombre_columna_tiempo_trabajo = "tiempo_trabajo_horas_" + numero_scanner_linea

    # Preparar dataset para esa linea
    df_prophet_linea = df_linea_trabajo_semanal_conjunto.rename(
        columns={"semana_primer_dia": "ds", nombre_columna_tiempo_trabajo: "y"})
    df_prophet_linea = df_prophet_linea[['ds', 'y']]

    # Predeciremos 4 meses (16 semanaas, por tanto se testean 16 semanas)
    fecha_max = df_linea_trabajo_semanal_conjunto['semana_primer_dia'].max()
    fecha_limite = fecha_max - pd.Timedelta(weeks=15)
    df_prophet_linea_train = df_prophet_linea[df_prophet_linea['ds'] < pd.to_datetime(fecha_limite)]
    df_prophet_linea_test = df_prophet_linea[df_prophet_linea['ds'] >= pd.to_datetime(fecha_limite)]

    # Preparamos modelo y preecimos

    model = Prophet(changepoint_prior_scale=3, interval_width=0.8)
    model.fit(df_prophet_linea_train)
    future = model.make_future_dataframe(periods=len(df_prophet_linea_test), freq='W-MON')
    forecast = model.predict(future)

    # Preparar dataset de comparación
    y_pred_prophet = forecast[['ds', 'yhat']].tail(len(df_prophet_linea_test))
    df_forecast_comparison_prophet = y_pred_prophet.merge(df_prophet_linea_test, on='ds')
    df_forecast_comparison_prophet['error'] = df_forecast_comparison_prophet['y'] - df_forecast_comparison_prophet[
        'yhat']

    RMSE = mean_squared_error(df_forecast_comparison_prophet['y'], df_forecast_comparison_prophet['yhat']) ** (1 / 2)
    print('RMSE Prophet: ' + str(RMSE))

    MAE = mean_absolute_error(df_forecast_comparison_prophet['y'], df_forecast_comparison_prophet['yhat'])
    print('MAE Prophet: ' + str(MAE))

    resultado_prophet = pd.DataFrame({
        'linea': [numero_scanner_linea],
        'RMSE': [RMSE],
        'MAE': [MAE],
    })

    precision_Prophet = pd.concat([precision_Prophet, resultado_prophet], ignore_index=True)

    # Entrenar modelo completo

    model = Prophet(changepoint_prior_scale=3, interval_width=0.8)

    model.fit(df_prophet_linea)

    # Realizar una predicción para el futuro
    future = model.make_future_dataframe(periods=16, freq='W-MON')
    forecast = model.predict(future)

    def limitar_cero(valor):
        if valor < 0:
            return 0
        else:
            return valor

    y_pred_prophet = forecast[['ds', 'yhat']].tail(len(df_prophet_linea_test))

    y_pred_prophet['yhat'] = y_pred_prophet['yhat'].apply(limitar_cero)

    print(y_pred_prophet)

    predicciones_Prophet['ds'] = y_pred_prophet['ds']
    predicciones_Prophet[numero_scanner_linea] = y_pred_prophet['yhat']

predicciones_Prophet['tipo'] = 'prediccion'

columns_to_keep = ['semana_primer_dia'] + [col for col in df_linea_trabajo_semanal_conjunto.columns if col.startswith('tiempo_trabajo_horas_')]
renamed_columns = {'semana_primer_dia': 'ds'}
renamed_columns.update({col: col.replace('tiempo_trabajo_horas_', '') for col in columns_to_keep[1:]})

# Crear nuevo DataFrame
df_valores_historicos = df_linea_trabajo_semanal_conjunto[columns_to_keep].rename(columns=renamed_columns)
df_valores_historicos['tipo'] = 'historico'

result_Prohet = pd.concat([df_valores_historicos,predicciones_Prophet], ignore_index = False)

result_Prohet.to_csv('../data/result_Prophet.csv', sep=';')
precision_Prophet.to_csv('../data/precision_Prophet.csv', sep=';')

print('Modelo Prophet Entrenado')

## 4: Pronóstico con Exponential Smoothing
#========================================================================================================================================
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.dates as mdates

def limitar_cero(valor):
    if valor < 0:
        return 0
    else:
        return valor

precision_Exp = pd.DataFrame()
predicciones_Exp = pd.DataFrame()

for numero_scanner_linea in numeros_scanners:

    print(numero_scanner_linea)

    # Preparamos datos.
    if numero_scanner_linea == 'SAPRFMADR15' or numero_scanner_linea == 'SAPRFMADR17':
        continue

    nombre_columna_tiempo_trabajo = "tiempo_trabajo_horas_" + numero_scanner_linea

    df_exp_linea = df_linea_trabajo_semanal_conjunto.rename(columns={"semana_primer_dia": "ds", nombre_columna_tiempo_trabajo: "y"})
    df_exp_linea = df_exp_linea[['ds', 'y']]

    fecha_max = df_exp_linea['ds'].max()

#Calcular métricas de error
    fecha_limite = fecha_max - pd.Timedelta(weeks=15)
    df_exp_linea_train = df_exp_linea[df_exp_linea['ds'] < pd.to_datetime(fecha_limite)]
    df_exp_linea_test = df_exp_linea[df_exp_linea['ds'] >= pd.to_datetime(fecha_limite)]

    model = ExponentialSmoothing(
        df_exp_linea_train['y'], 
        trend='add', #mul
        seasonal='add', #mul
        seasonal_periods=52
    ).fit(
        optimized=True  # No optimizar automáticamente
        )

    y_pred_exp_linea_test = model.forecast(16)

    RMSE = mean_squared_error(df_exp_linea_test['y'], y_pred_exp_linea_test) ** (1 / 2)
    print('RMSE Exponential Smoothing: ' + str(RMSE))

    MAE = mean_absolute_error(df_exp_linea_test['y'], y_pred_exp_linea_test)
    print('MAE Exponential Smoothing: ' + str(MAE))

    resultado_exp = pd.DataFrame({
        'linea': [numero_scanner_linea],
        'RMSE': [RMSE],
        'MAE': [MAE],
    })

    precision_Exp = pd.concat([precision_Exp, resultado_exp], ignore_index=True)
    
#Predecir valores futuros
    print("Prediciendo Valores Futuros Exponential Smoothing")
    last_date = fecha_max
    future_dates = pd.date_range(start=last_date, periods=17, freq='W-MON')[1:]

    model = ExponentialSmoothing(
        df_exp_linea['y'], 
        trend='add', #mul
        seasonal='add', #mul
        seasonal_periods=52
    ).fit(
        optimized=True  # No optimizar automáticamente
        )

    y_pred_exp_linea = model.forecast(16)

    data = {
    'ds': future_dates,
    'yhat': y_pred_exp_linea
    }

    y_pred_exp_linea = pd.DataFrame(data)

    y_pred_exp_linea['yhat'] = y_pred_exp_linea['yhat'].apply(limitar_cero)

    predicciones_Exp['ds'] = y_pred_exp_linea['ds']
    predicciones_Exp[numero_scanner_linea] = y_pred_exp_linea['yhat']

predicciones_Exp['tipo'] = 'prediccion'

columns_to_keep = ['semana_primer_dia'] + [col for col in df_linea_trabajo_semanal_conjunto.columns if col.startswith('tiempo_trabajo_horas_')]
renamed_columns = {'semana_primer_dia': 'ds'}
renamed_columns.update({col: col.replace('tiempo_trabajo_horas_', '') for col in columns_to_keep[1:]})

# Crear nuevo DataFrame
df_valores_historicos = df_linea_trabajo_semanal_conjunto[columns_to_keep].rename(columns=renamed_columns)
df_valores_historicos['tipo'] = 'historico'

result_Exp = pd.concat([df_valores_historicos,predicciones_Exp], ignore_index = False)

result_Exp.to_csv('../data/result_Exp.csv', sep=';')
precision_Exp.to_csv('../data/precision_Exp.csv', sep=';')


## 5: Pronóstico con SARIMA
#========================================================================================================================================
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

best_pdq = (3, 2, 4)
best_seasonal_pdq = (2, 0, 2, 52)

def limitar_cero(valor):
    if valor < 0:
        return 0
    else:
        return valor

precision_Sarima = pd.DataFrame()
predicciones_Sarima = pd.DataFrame()

for numero_scanner_linea in numeros_scanners:

    print(numero_scanner_linea)

    # Preparamos datos.
    if numero_scanner_linea == 'SAPRFMADR15' or numero_scanner_linea == 'SAPRFMADR17':
        continue

    nombre_columna_tiempo_trabajo = "tiempo_trabajo_horas_" + numero_scanner_linea

    df_sarima_linea = df_linea_trabajo_semanal_conjunto.rename(columns={"semana_primer_dia": "ds", nombre_columna_tiempo_trabajo: "y"})
    df_sarima_linea = df_sarima_linea[['ds', 'y']]

    fecha_max = df_sarima_linea['ds'].max()
    
#Predecir valores futuros
    print("Prediciendo Valores Futuros SARIMA")

    auto_model = auto_arima(
    df_sarima_linea['y'],
    seasonal=True, 
    m=52, 
    trace=True,
    error_action='ignore',
    suppress_warnings=True, 
    stepwise=True
    )
    
    best_pdq = auto_model.order
    best_seasonal_pdq = auto_model.seasonal_order

    last_date = fecha_max
    future_dates = pd.date_range(start=last_date, periods=17, freq='W-MON')[1:]

    forecaster = SARIMAX(
        df_sarima_linea['y'],
        order=best_pdq,
        seasonal_order=best_seasonal_pdq,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = forecaster.fit(maxiter=1000)

    y_pred_sarima_linea = results.get_forecast(steps=16).predicted_mean

    data = {
    'ds': future_dates,
    'yhat': y_pred_sarima_linea
    }

    y_pred_sarima_linea = pd.DataFrame(data)

    y_pred_sarima_linea['yhat'] = y_pred_sarima_linea['yhat'].apply(limitar_cero)

    predicciones_Sarima['ds'] = y_pred_sarima_linea['ds']
    predicciones_Sarima[numero_scanner_linea] = y_pred_sarima_linea['yhat']

#Calcular métricas de error
    fecha_limite = fecha_max - pd.Timedelta(weeks=15)
    df_sarima_linea_train = df_sarima_linea[df_sarima_linea['ds'] < pd.to_datetime(fecha_limite)]
    df_sarima_linea_test = df_sarima_linea[df_sarima_linea['ds'] >= pd.to_datetime(fecha_limite)]
    
    forecaster = SARIMAX(
        df_sarima_linea_train['y'],
        order=best_pdq,
        seasonal_order=best_seasonal_pdq,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = forecaster.fit(maxiter=1000)

    y_pred_sarima_linea_test = results.get_forecast(steps=16).predicted_mean

    RMSE = mean_squared_error(df_sarima_linea_test['y'], y_pred_sarima_linea_test) ** (1 / 2)
    print('RMSE Exponential Smoothing: ' + str(RMSE))

    MAE = mean_absolute_error(df_sarima_linea_test['y'], y_pred_sarima_linea_test)
    print('MAE Exponential Smoothing: ' + str(MAE))

    resultado_sarima = pd.DataFrame({
        'linea': [numero_scanner_linea],
        'RMSE': [RMSE],
        'MAE': [MAE],
    })

    precision_Sarima = pd.concat([precision_Sarima, resultado_sarima], ignore_index=True)

predicciones_Sarima['tipo'] = 'prediccion'

columns_to_keep = ['semana_primer_dia'] + [col for col in df_linea_trabajo_semanal_conjunto.columns if col.startswith('tiempo_trabajo_horas_')]
renamed_columns = {'semana_primer_dia': 'ds'}
renamed_columns.update({col: col.replace('tiempo_trabajo_horas_', '') for col in columns_to_keep[1:]})

# Crear nuevo DataFrame
df_valores_historicos = df_linea_trabajo_semanal_conjunto[columns_to_keep].rename(columns=renamed_columns)
df_valores_historicos['tipo'] = 'historico'

result_Sarima = pd.concat([df_valores_historicos,predicciones_Sarima], ignore_index = False)

result_Sarima.to_csv('../data/result_Sarima.csv', sep=';')
precision_Sarima.to_csv('../data/precision_Sarima.csv', sep=';')

## 6: Pronóstico con XGBoost
#========================================================================================================
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

precision_XGBoost = pd.DataFrame()
predicciones_XGBoost = pd.DataFrame()


for numero_scanner_linea in numeros_scanners:

    print(numero_scanner_linea)

    # Preparamos datos.

    if numero_scanner_linea == 'SAPRFMADR15' or numero_scanner_linea == 'SAPRFMADR17':
        continue

    nombre_columna_tiempo_trabajo = "tiempo_trabajo_horas_" + numero_scanner_linea

    # Preparar dataset para esa linea
    #Rellenar semanas faltantes

    semana_min = df_linea_trabajo_semanal_conjunto['semana_primer_dia'].min()
    semana_max = df_linea_trabajo_semanal_conjunto['semana_primer_dia'].max()
    todas_las_semanas = pd.date_range(semana_min, semana_max, freq='W-MON')
    todas_las_semanas
    df_todas_las_semanas = pd.DataFrame({'semana_primer_dia': todas_las_semanas})
    df_completo = pd.merge(df_todas_las_semanas, df_linea_trabajo_semanal_conjunto, on='semana_primer_dia', how='left')
    df_completo[nombre_columna_tiempo_trabajo].fillna(0, inplace=True)
    df_completo['semana'] = df_completo['semana_primer_dia'].dt.isocalendar().week.astype(str) + "-" + df_completo['semana_primer_dia'].dt.year.astype(str)
    df_linea_trabajo_semanal_conjunto_xg = df_completo

    df_xg = df_linea_trabajo_semanal_conjunto_xg.rename(columns={"semana_primer_dia": "ds", nombre_columna_tiempo_trabajo: "y"})
    df_xg = df_xg[['ds', 'y']]

    df_xg['week'] = df_xg['ds'].dt.isocalendar().week
    df_xg['year'] = df_xg['ds'].dt.isocalendar().year
    df_xg['month'] = df_xg['ds'].dt.month
    
    def is_holiday(y):
        # Si y no es None, aplicar el criterio original
        if y < 5:
            return 1
        else:
            return 0
    
    def is_holiday_future(ds):
        week = ds.isocalendar().week
        month = ds.month
        
        # Verificar si la semana está en las semanas 32 o 33 de agosto
        if (week == 33 or week == 34) and month == 8:
            return 1
        else:
            return 0
    
    df_xg['holiday'] = df_xg.apply(lambda row: is_holiday(row['y']), axis=1)
    
    df_xg.set_index('ds', inplace = True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_xg['y'].values.reshape(-1, 1))
    
    sequence_length = 52  # 52 semanas, año completo HIPERPARÁMETRO
    
    #Generar los datos de entrenamiento a partir de la serie temporal completa escalada
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
    
        data_append_X = np.concatenate([scaled_data[i-sequence_length:i, 0], df_xg.iloc[i][['year', 'month','week', 'holiday']].values])
        
        X.append(data_append_X)
    
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)

    #Test Size para que haya mismos datos de entrenamiento que en los otros
    
    df_X_xg = pd.DataFrame(X)
    df_y_xg = pd.DataFrame(y)
    
    df_X_xg = df_X_xg.apply(pd.to_numeric, errors='coerce')
    df_y_xg = df_y_xg.apply(pd.to_numeric, errors='coerce')
    
    X_train, X_test, y_train, y_test = train_test_split(df_X_xg, df_y_xg, test_size=0.12, shuffle=False, random_state=101)
    
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.1,  # Ajustado
        max_depth=10,         # Ajustado
        subsample=0.7,       # Ajustado
        colsample_bytree=0.9, # Ajustado
        n_estimators=500,    # Ajustado
        reg_alpha=0,       # Regularización L1
        reg_lambda=1,       # Regularización L2
        early_stopping_rounds= 50
        )

    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=True,
    )

    y_pred_xg = model.predict(X_test)
    y_test_inverted = scaler.inverse_transform(y_test.values)
    y_pred_xg_inverted = scaler.inverse_transform(y_pred_xg.reshape(-1, 1))

    error = pd.DataFrame(y_test_inverted-y_pred_xg_inverted)
    desviacion_error = error.mean()
    desviacion_error = float(desviacion_error.iloc[0])

    #Compensar los valores pronosticados y recalcular:
    y_pred_xg_inverted_compensado = y_pred_xg_inverted + desviacion_error

    RMSE = mean_squared_error(y_test_inverted, y_pred_xg_inverted_compensado) ** (1 / 2)
    print('RMSE XGBoost: ' + str(RMSE))

    MAE = mean_absolute_error(y_test_inverted, y_pred_xg_inverted_compensado)
    print('MAE Prophet: ' + str(MAE))

    resultado_XGBoost = pd.DataFrame({
        'linea': [numero_scanner_linea],
        'RMSE': [RMSE],
        'MAE': [MAE],
    })

    precision_XGBoost = pd.concat([precision_XGBoost, resultado_XGBoost], ignore_index=True)


    #Entrenar modelo con todos los datos:
    
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.1,  # Ajustado
        max_depth=10,         # Ajustado
        subsample=0.7,       # Ajustado
        colsample_bytree=0.9, # Ajustado
        n_estimators=500,    # Ajustado
        reg_alpha=0,       # Regularización L1
        reg_lambda=1,       # Regularización L2
        # # early_stopping_rounds= 50
        )
    
    model.fit(
        df_X_xg, 
        df_y_xg, 
        verbose=True,
    )


    # Número de predicciones futuras que quieres hacer
    n_pred = 16  # Por ejemplo, predecir las próximas 16 semanas
    
    # Crear un DataFrame para almacenar las predicciones futuras
    pred_futuras = pd.DataFrame(index=range(n_pred), columns=df_xg.columns)
    
    # # Inicializar con la última secuencia de datos disponible
    last_sequence = scaled_data[-sequence_length:]
    
    # Modificar la lógica de predicción para incluir la nueva condición de 'holiday'
    for i in range(n_pred):
        # Calcular la fecha de la predicción actual
        current_date = df_xg.index[-1] + pd.Timedelta(weeks=i+1)
    
        # Extraer el año, mes, semana y calcular si es un feriado
        current_year = current_date.year
        current_month = current_date.month
        current_week = current_date.isocalendar().week
        current_holiday = is_holiday_future(current_date) 
    
        # Concatenar la secuencia de datos con las características temporales actualizadas
        input_sequence = np.concatenate([last_sequence.flatten(), [current_year, current_month, current_week, current_holiday]])
    
        input_sequence = input_sequence.reshape(1, -1)
        
        next_prediction = model.predict(input_sequence)
        next_prediction_inverted = scaler.inverse_transform(next_prediction.reshape(-1, 1))[0][0]
    
        # Guardar los resultados en el DataFrame de predicciones futuras
        pred_futuras.loc[i, 'y'] = next_prediction_inverted
        pred_futuras.loc[i, 'ds'] = current_date
        pred_futuras.loc[i, 'year'] = current_year
        pred_futuras.loc[i, 'month'] = current_month
        pred_futuras.loc[i, 'week'] = current_week
        pred_futuras.loc[i, 'holiday'] = current_holiday
    
        # Actualizar la secuencia para la próxima predicción
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_prediction
    
    # Imprimir las predicciones futuras
    print(pred_futuras)
    pred_futuras_compensadas = pred_futuras.copy()
    pred_futuras_compensadas['y'] = pred_futuras_compensadas['y'] + desviacion_error

    def limitar_cero(valor):
        return max(0, valor)
        
    pred_futuras_compensadas['y'] = pred_futuras_compensadas['y'].apply(limitar_cero)

    y_pred_XGBoost = pred_futuras_compensadas[['ds', 'y']].rename(columns={'y': 'yhat'})

    
    predicciones_XGBoost['ds'] = y_pred_XGBoost['ds']
    predicciones_XGBoost[numero_scanner_linea] = y_pred_XGBoost['yhat']

    predicciones_XGBoost['tipo'] = 'prediccion'

    columns_to_keep = ['semana_primer_dia'] + [col for col in df_linea_trabajo_semanal_conjunto.columns if col.startswith('tiempo_trabajo_horas_')]
    renamed_columns = {'semana_primer_dia': 'ds'}
    renamed_columns.update({col: col.replace('tiempo_trabajo_horas_', '') for col in columns_to_keep[1:]})
    
    # Crear nuevo DataFrame
    df_valores_historicos = df_linea_trabajo_semanal_conjunto_xg[columns_to_keep].rename(columns=renamed_columns)
    df_valores_historicos['tipo'] = 'historico'
    
    result_XGBoost = pd.concat([df_valores_historicos,predicciones_XGBoost], ignore_index = False)
    
    result_XGBoost.to_csv('../data/result_XGBoost.csv', sep=';')
    precision_XGBoost.to_csv('../data/precision_XGBoost.csv', sep=';')
    
    print('Modelo XGBoost Entrenado')