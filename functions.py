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
    