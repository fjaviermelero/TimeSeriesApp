import streamlit as st
import pandas as pd
import plotly.express as px

# Cargar los archivos CSV
@st.cache_data
def load_data():
    # Carga los datos de los modelos
    data_exp = pd.read_csv('../data/result_Exp.csv', sep=';')
    data_prophet = pd.read_csv('../data/result_Prophet.csv', sep=';')
    data_sarima = pd.read_csv('../data/result_Sarima.csv', sep=';')
    data_xgboost = pd.read_csv('../data/result_XGBoost.csv', sep=';')

    precision_exp = pd.read_csv('../data/precision_Exp.csv', sep=';')
    precision_prophet = pd.read_csv('../data/precision_Prophet.csv', sep=';')
    precision_sarima = pd.read_csv('../data/precision_Sarima.csv', sep=';')
    precision_xgboost = pd.read_csv('../data/precision_XGBoost.csv', sep=';')

    # Limpiar datos si es necesario
    for df in [data_exp, data_prophet, data_sarima, data_xgboost]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
    
    return data_exp, precision_exp, data_prophet, precision_prophet, data_sarima, precision_sarima, data_xgboost, precision_xgboost

# Logo de la empresa
st.image('../data/steelcase-logo.png', width=200)

# Título de la aplicación
st.title("Planta Madrid")

# Cargar los datos
data_exp, precision_exp, data_prophet, precision_prophet, data_sarima, precision_sarima, data_xgboost, precision_xgboost = load_data()

# Diccionario de mapeo de nombres de columnas
column_mapping = {
    "total": "Planta Completa",
    "SAPRFMADR26": "Mesas",
    "SAPRFMADR20": "Paneles",
    "SAPRFMADR23": "Tableros",
    "SAPRFMADR16": "Volúmenes",
    "SAPRFMADR19": "Conectores",
    "SAPRFMADR21": "Varios",
    "SAPRFMADR17": "Silla DO",
    "SAPRFMADR18": "Pie Ajustable",
    "SAPRFMADR24": "Carpinteria otros",
    "SAPRFMADR25": "Partito Rail"
}

# Obtener solo las claves del diccionario
columnas_nombre_linea = list(column_mapping.values())

# Seleccionar una columna para graficar usando nombres amigables
selected_column_name = st.selectbox('Selecciona la línea que quieres visualizar', columnas_nombre_linea)

# Obtener la clave correspondiente a la columna seleccionada
selected_column = list(column_mapping.keys())[list(column_mapping.values()).index(selected_column_name)]

# Convertir las columnas 'ds' a tipo datetime en cada DataFrame
for data in [data_exp, data_prophet, data_sarima, data_xgboost]:
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')

# Verificar si alguna tabla tiene fechas no válidas
for name, df in zip(['Exponencial', 'Prophet', 'SARIMA', 'XGBoost'], [data_exp, data_prophet, data_sarima, data_xgboost]):
    if df['ds'].isnull().any():
        st.warning(f"Se encontraron fechas no válidas en la columna 'ds' del modelo {name}.")

# Agregar botones para seleccionar el modelo
st.subheader("Selecciona el modelo a visualizar")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    btn_exp = st.button('Exponencial')
with col2:
    btn_prophet = st.button('Prophet')
with col3:
    btn_sarima = st.button('SARIMA')
with col4:
    btn_xgboost = st.button('XGBoost')
with col5:
    btn_all = st.button('Todos')

# Configurar qué modelos mostrar según el botón seleccionado
if btn_exp or not any([btn_exp, btn_prophet, btn_sarima, btn_xgboost, btn_all]):
    models_to_display = ['Exponencial']
elif btn_prophet:
    models_to_display = ['Prophet']
elif btn_sarima:
    models_to_display = ['SARIMA']
elif btn_xgboost:
    models_to_display = ['XGBoost']
elif btn_all:
    models_to_display = ['Exponencial', 'Prophet', 'SARIMA', 'XGBoost']

# Combinar datos para graficar
data_exp['modelo'] = 'Exponencial'
data_prophet['modelo'] = 'Prophet'
data_sarima['modelo'] = 'SARIMA'
data_xgboost['modelo'] = 'XGBoost'

combined_data = pd.concat([data_exp, data_prophet, data_sarima, data_xgboost])


# Filtrar datos por la columna seleccionada y los modelos a mostrar
# Separar la serie histórica y las predicciones
if btn_all:
    # Filtrar la serie histórica solo del modelo 'Exponencial'
    historical_data = combined_data[
        (combined_data['modelo'] == 'Exponencial') & 
        (combined_data['tipo'] == 'historico')  # Asegúrate de tener esta distinción en tus datos
    ]

    # Filtrar las predicciones de todos los modelos
    prediction_data = combined_data[
        (combined_data['modelo'].isin(models_to_display)) & 
        (combined_data['tipo'] != 'historico')  # Excluir la serie histórica para otros modelos
    ]

    # Combinar los datos para la gráfica
    filtered_data = pd.concat([historical_data, prediction_data])
else:
    # Filtrar datos por la columna seleccionada y los modelos a mostrar
    filtered_data = combined_data[
        (combined_data['modelo'].isin(models_to_display)) & 
        (combined_data[selected_column].notnull())
    ]


# Obtener métricas para la columna seleccionada
metrics = pd.concat([
    precision_exp[precision_exp['linea'] == selected_column].assign(modelo='Exponencial'),
    precision_prophet[precision_prophet['linea'] == selected_column].assign(modelo='Prophet'),
    precision_sarima[precision_sarima['linea'] == selected_column].assign(modelo='SARIMA'),
    precision_xgboost[precision_xgboost['linea'] == selected_column].assign(modelo='XGBoost')
], ignore_index=True)

# Limpiar columnas no deseadas en 'metrics'
columns_to_keep = ['modelo', 'linea', 'RMSE', 'MAE']  # Incluye solo las necesarias
metrics = metrics[columns_to_keep]

# Mostrar métricas en una tabla
st.subheader("Métricas de Error")
if not metrics.empty:
    metrics_filtered = metrics[metrics['modelo'].isin(models_to_display)][['modelo', 'RMSE', 'MAE']]
    metrics_filtered = metrics_filtered.rename(columns={'modelo': 'Modelo', 'RMSE': 'Error RMSE', 'MAE': 'Error MAE'})
    metrics_filtered = metrics_filtered.reset_index(drop=True)  # Elimina el índice
    st.dataframe(metrics_filtered, use_container_width=True, hide_index=True)
else:
    st.warning("No se encontraron métricas para los modelos seleccionados.")

# Gráfica interactiva con los modelos seleccionados
fig = px.line(
    filtered_data,
    x='ds',
    y=selected_column,
    color='modelo',
    line_dash='tipo',
    color_discrete_map={
        'Exponencial': 'green',
        'Prophet': 'red',
        'SARIMA': 'orange',
        'XGBoost': 'purple'
    }
)

# Sobrescribir colores para asegurarse de que 'historico' siempre sea azul
for i, d in enumerate(fig.data):
    if 'historico' in d.legendgroup:
        fig.data[i].line.color = 'blue'

fig.update_layout(
    title="Comparación de Predicciones por Modelo",
    xaxis_title='Fecha',
    yaxis_title='Horas Efectivas',
    xaxis_tickangle=-45
)

st.plotly_chart(fig)
