import pandas as pd
import numpy as np
import warnings
import datetime


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

# Función agrega columnas de pips
def f_columnas_pips(param_data: pd.DataFrame) -> pd.DataFrame:
    # Transponer para usar .loc
    param_data = param_data.T
    # Crear lista que tendra la informacion de los pips
    pips = []
    
    for position in range(len(param_data.T)):
        # Obtener multiplicador de la postura que estamos viendo
        multiplier = f_pip_size(param_data.loc["Symbol"][position])
        # Obtener pips
        if param_data.loc["Type"][position] == "buy":
            pips.append((param_data.loc["Closeprice"][position] - param_data.loc["Openprice"][position]) * multiplier)
        else:
            pips.append((param_data.loc["Openprice"][position] - param_data.loc["Closeprice"][position]) * multiplier)

   # Agregar columnas a df
    param_data = param_data.T
    param_data['pips'] = pips
    param_data['pips_acm'] = param_data['pips'].cumsum()
    param_data['profit_acm'] = param_data['Profit'].cumsum()
    
    return param_data


# Funcion para obtener metricas de cuenta de trading
def f_estadisticas_ba(param_data: pd.DataFrame) -> dict:
    
    ######## CREAR DF 1
    # Hacer nuevos dfs uno de operaciones largas y otro de cortas
    compras = param_data[param_data.T.loc["Type"] == "buy"]
    ventas = param_data[param_data.T.loc["Type"] == "sell"]

   # Crear tabla con metricas
    df_1_tabla = pd.DataFrame(index=["medida", "valor", "descripcion"])
    df_1_tabla["1"] = ["Ops totales", len(param_data), "Operaciones totales"]
    df_1_tabla["2"] = ["Ganadoras", sum([1 for i in param_data.T.loc["Profit"] if i > 0]), "Operaciones ganadoras"]
    df_1_tabla["3"] = ["Ganadoras_c",  sum([1 for i in compras.T.loc["Profit"] if i > 0]), "Operaciones ganadoras de compra"]
    df_1_tabla["4"] = ["Ganadoras_v",  sum([1 for i in ventas.T.loc["Profit"] if i > 0]), "Operaciones ganadoras de venta"]
    df_1_tabla["5"] = ["Perdedoras", sum([1 for i in param_data.T.loc["Profit"] if i < 0]), "Operaciones perdedoras"]
    df_1_tabla["6"] = ["Perdedoras_c",  sum([1 for i in compras.T.loc["Profit"] if i < 0]), "Operaciones perdedoras de compra"]
    df_1_tabla["7"] = ["Perdedoras_v",  sum([1 for i in ventas.T.loc["Profit"] if i < 0]), "Operaciones perdedoras de venta"]
    df_1_tabla["8"] = ["Mediana (Profit)", np.percentile(param_data.T.loc["Profit"], 50), "Mediana de profit de operaciones"]
    df_1_tabla["9"] = ["Mediana (Pips)", np.percentile(param_data.T.loc["pips"], 50), "Mediana de pips de operaciones"]
    df_1_tabla["10"] = ["r_efectividad", sum([1 for i in param_data.T.loc["Profit"] if i > 0])/len(param_data), "Ganadoras Totales/Operaciones Totales"]
    df_1_tabla["11"] = ["r_proporcion", sum([1 for i in param_data.T.loc["Profit"] if i > 0])/sum([1 for i in param_data.T.loc["Profit"] if i < 0]), "Ganadoras Totales/Perdedoras Totales"]
    df_1_tabla["12"] = ["r_efectividad_c", sum([1 for i in compras.T.loc["Profit"] if i > 0])/len(compras), "Ganadoras Compras/Operaciones Totales"]
    df_1_tabla["13"] = ["r_efectividad_v", sum([1 for i in ventas.T.loc["Profit"] if i > 0])/len(ventas), "Ganadoras Ventas/ Operaciones Totales"]
   # Transponer
    df_1_tabla = df_1_tabla.T
    
    ########## CREAR DF 2
    # Crear set con valores unicos de las monedas tradeadas
    unique_instruments = set(param_data.T.loc["Symbol"])

    # Crear lista a rellenar
    performance_list = []

    # Iterar cada instrumento para encontrar su ratio de efectividad
    for sym in unique_instruments:
        # DF temporal con x insturmento filtrado
        temp_param_data = param_data[param_data.T.loc["Symbol"] == sym]
        # Encontrar performance de instrumento
        performance_list.append(
            round(sum([1 for i in temp_param_data.T.loc["Profit"] if i > 0]) / len(temp_param_data) * 100, 2))

    # Crear df de ranking
    df_2_ranking = pd.DataFrame()
    df_2_ranking["symbol"] = list(unique_instruments)
    df_2_ranking["rank"] = performance_list
    # Ordenar de mayor a menor
    df_2_ranking.sort_values("rank", inplace=True, ascending=False)

    ########### REGREASAR DF con diccionario
    return {"df_1_tabla": df_1_tabla, "df_2_ranking": df_2_ranking}

# Función evolucion del capital
def f_evolucion_capital(param_data: pd.DataFrame):
    # Cambiar formato de columnas para no aparezca hora, solo fecha
    param_data["Opentime"] = param_data["Opentime"].apply(lambda x: x.date()) if type(param_data.iloc[0,0]) != datetime.date \
                                                                              else param_data["Opentime"]
    # Hacer pivote para dejar la suma del profit por fecha
    pivot = param_data.pivot_table(values="Profit", index="Opentime", aggfunc=sum).reset_index()
    # Renombrar columnas
    pivot.columns = ["timestamp", "profit_d"]
    # Hacer suma acumulada de profit diario
    pivot["profit_d_acum"] = pivot["profit_d"].cumsum() + 100000
    
    return pivot