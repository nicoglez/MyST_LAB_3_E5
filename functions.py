import pandas as pd
import numpy as np
import warnings


# Archivo para leer xlsx de MT5 o MT4
def f_leer_archivo(param_archivo: str) -> pd.DataFrame:
    
    # Nombre de columnas en Ingles
    col_names = ['Opentime', 'Position', 'Symbol', 'Type', 'Volume', 'Openprice', 'StopLoss', 'TakeProfit', 
                'Closetime', 'Closeprice', 'Commission', 'Swap', 'Profit', 'Unnamed: 13', 'Unnamed: 14']
    
    # Leer archivo, ignorando warnings que filtraremos posteriormente
    with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            data = pd.read_excel(param_archivo, engine="openpyxl", skiprows=6)
    # Renombrar columnas si no estan en         
    data.rename(columns=dict(zip(data.columns.values, col_names)), inplace=True) 
    # Quitar rows que no necesitamos, quitando aquellas que no afectan la posicion, de ordenes para adelante (español o ingles)
    # Metratrader 5
    if np.argmax(data.iloc[:, 0].values == "Orders") != 0 or np.argmax(data.iloc[:, 0].values == "Órdenes") != 0:
        data = data.iloc[:np.argmax(data.iloc[:, 0].values == "Orders"), 0:-2] \
            if sum(data.iloc[:, 0].values == "Orders") == 1 \
            else data.iloc[:np.argmax(data.iloc[:, 0].values == "Órdenes"), 0:-2]

    # Metatrader 4
    else:
        data = data.iloc[:-sum([type(i) == float for i in data.iloc[:, 0].values]), 0:-2]

    # Regresar DataFrame
    return data


# Funcion para obtener multiplicador de pips
def f_pip_size(param_ins: str) -> float:
    # Leer archivo con informacion de los pips
    pip_data = pd.read_csv("files/instruments_pips.csv")
    # Hacer tickets todos mayusculas para comparar params en mayusculas
    pip_data.iloc[:, 1] = [pip.upper() for pip in pip_data.iloc[:, 1]]
    # Hacer params mayusculas
    param_ins = param_ins.upper()

    # Ver si es divisa para añadir /
    if len(param_ins) == 6:
        param_ins = param_ins[0:3] + "/" + param_ins[3:]

    # Obtener informacion de pips del csv, regresar multiplicador 100 si no hay datos
    return 100 if sum(pip_data.iloc[:, 1] == param_ins) == 0 else 1/pip_data.iloc[np.argmax(pip_data.iloc[:, 1] == param_ins), 3]

# Funcion que crea la columna tiempo
def f_columnas_tiempos(param_data: pd.DataFrame):
    data = param_data
    # Convertimos fechas de apertura y cierre para encontrar la diferencia y saber cuanto duró abierta la posición
    data["tiempo"] = (pd.to_datetime(data['Closetime']) - pd.to_datetime(data['Opentime'])).astype('timedelta64[s]')
    # Convertimos "Opentime" y "Closetime" a tipo datetime 
    data["Opentime"] = pd.to_datetime(data['Opentime'])
    data["Closetime"] = pd.to_datetime(data['Closetime'])
    return data