import pandas as pd
import numpy as np
import warnings


# Archivo para leer xlsx de MT5
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
    data = data.iloc[:np.argmax(data.iloc[:, 0].values == "Orders"), 0:-3] \
                    if sum(data.iloc[:, 0].values == "Orders")==1 \
                    else data.iloc[:np.argmax(data.iloc[:, 0].values == "Órdenes"), 0:-3]
    # Quitar blanks: comisiones y swaps`
    data = data[data.columns.drop(["Commission", "Swap"])]
        
        
    # Regresar DataFrame
    return data


# Funcion para obtener multiplicador de pips
def f_pip_size(param_ins: str) -> float:
    # Leer archivo con informacion de los pips
    pip_data = pd.read_csv("files/instruments_pips.csv")

    # Ver si es divisa para añadir /
    if len(param_ins) == 6:
        param_ins = param_ins[0:3] + "/" + param_ins[3:]

        # Obtener informacion de pips del csv, regresar multiplicador 100 si no hay datos
    return 100 if sum(pip_data.iloc[:, 1] == param_ins) == 0 else 1/pip_data.iloc[np.argmax(pip_data.iloc[:, 1] == param_ins), 3]
