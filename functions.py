import pandas as pd
import pandas_datareader.data as web
import numpy as np
import warnings
import datetime
from typing import Optional


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
    param_data["by_day"] = param_data["Opentime"]
    param_data["by_day"] = param_data["by_day"].apply(lambda x: x.date()) if type(
        param_data.iloc[0, 0]) != datetime.date \
        else param_data["by_day"]
    # Hacer pivote para dejar la suma del profit por fecha
    pivot = param_data.pivot_table(values="Profit", index="by_day", aggfunc=sum).reset_index()
    # Renombrar columnas
    pivot.columns = ["timestamp", "profit_d"]
    # Hacer suma acumulada de profit diario
    pivot["profit_d_acum"] = pivot["profit_d"].cumsum() + 100000

    return pivot

# Función para descargar precio o precios de cierre
def get_adj_closes(tickers: str, start_date: str = None, end_date: Optional[str] = None, freq: str = 'd'):
    # Bajar solo un dato si end_date no se da
    end_date = end_date if end_date else start_date or None
    # Bajar cierre de precio ajustado
    closes = web.YahooDailyReader(symbols=tickers, start=start_date, end=end_date, interval=freq).read()['Adj Close']
    # Poner indice de forma ascendente
    closes.sort_index(inplace=True)
    return closes

# Funcion para obtener metricas de nuestra estrategia
def f_estadisticas_mad(evolucion):
    # Obtener metricas del portafolio
    mean_log_portafolio = np.log(evolucion["profit_d_acum"] / evolucion["profit_d_acum"].shift()).dropna().mean()
    rf = .05 / 365 * len(evolucion)
    sdport = np.log(evolucion["profit_d_acum"] / evolucion["profit_d_acum"].shift()).dropna().std()
    if len(np.log(evolucion["profit_d_acum"] / evolucion["profit_d_acum"].shift()).dropna()) == 1:
        sdport = 0

    # Obtener datos historicos del SP500 y calcular sus metricas
    SP_data = pd.DataFrame([get_adj_closes(tickers="^GSPC", start_date=date)[0] for date in evolucion["timestamp"]])
    mean_log_SP = np.log(SP_data / SP_data.shift()).dropna().mean()
    sd_port_SP = (np.array(evolucion["profit_d_acum"]) - np.array(SP_data)).std()

    # Calcular sharpe original y actualizado
    sharpe_original = 0 if sdport == 0 else round((mean_log_portafolio - rf) / sdport, 5)
    sharpe_actualizado = round(float((mean_log_portafolio - mean_log_SP) / sd_port_SP), 5)

    # Calcular Drawdown y Fechas
    date_1_dd = evolucion.iloc[np.argmax(evolucion["profit_d_acum"]), 0]
    date_2_dd = evolucion.iloc[
        np.argmax(evolucion["profit_d_acum"] == min(evolucion.iloc[np.argmax(evolucion["profit_d_acum"]):, 2])), 0]
    drawn_down = round(
        min(evolucion.iloc[np.argmax(evolucion["profit_d_acum"]):, 2]) / max(evolucion["profit_d_acum"]) - 1, 5)

    # Calcular Drawnup y Fechas
    date_1_du = evolucion.iloc[np.argmin(evolucion["profit_d_acum"]), 0]
    date_2_du = evolucion.iloc[
        np.argmax(evolucion["profit_d_acum"] == max(evolucion.iloc[np.argmin(evolucion["profit_d_acum"]):, 2])), 0]
    drawn_up = round(
        max(evolucion.iloc[np.argmin(evolucion["profit_d_acum"]):, 2]) / min(evolucion["profit_d_acum"]) - 1, 5)

      # Armar df
    df = pd.DataFrame(index=("valor", "descripcion"))
    df["sharpe_original"] = [sharpe_original, "Sharpe Ratio Fórmula Original"]
    df["sharpe_actualizado"] = [sharpe_actualizado, "Sharpe Ratio Fórmula Ajustada"]
    df["drawndown_f1"] = [date_1_dd, "Fecha inicial del DrawDown de Capital"]
    df["drawndown_f2"] = [date_2_dd, "Fecha final del DrawDown de Capital"]
    df["drawndown_capi"] = [drawn_down, "Máxima pérdida flotante registrada"]
    df["drawnup_f1"] = [date_1_du, "Fecha inicial del DrawUp de Capital"]
    df["drawnup_f2"] = [date_2_du, "Fecha final del DrawUp de Capital"]
    df["drawnup_capi"] = [drawn_up, "Máxima ganancia flotante registrada"]
    
    return df.T
