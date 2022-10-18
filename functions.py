import pandas as pd
import pandas_datareader.data as web
import numpy as np
import warnings
import datetime
from typing import Optional
from datetime import datetime, timedelta
import MetaTrader5 as MT5
import pytz

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
        data = data.iloc[:np.argmax(data.iloc[:, 0].values == "Orders"), 0:-3] \
            if sum(data.iloc[:, 0].values == "Orders") == 1 \
            else data.iloc[:np.argmax(data.iloc[:, 0].values == "Órdenes"), 0:-3]

    # Metatrader 4
    else:
        data = data.iloc[:-sum([type(i) == float for i in data.iloc[:, 0].values]), 0:-3]

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

def get_MT5_price(symbol, start_date, end_date):
    # Inicializar MT5
    if not MT5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # Cambiar timezone a UTC
    timezone = pytz.timezone("Etc/UTC")

    # Crear fecha de inicio y final
    year_1, month_1, day_1 = start_date.year, start_date.month, start_date.day
    utc_from = datetime(year_1, month_1, day_1, tzinfo=timezone)
    year_2, month_2, day_2 = end_date.year, end_date.month, end_date.day
    utc_to = datetime(year_2, month_2, day_2, tzinfo=timezone)

    # get bars from USDJPY M5 within the interval of 2020.01.10 00:00 - 2020.01.11 13:00 in UTC time zone
    rates = MT5.copy_rates_range(symbol.upper(), MT5.TIMEFRAME_M1, utc_from, utc_to)
    # cerrar conexion
    MT5.shutdown()

    # crear DF
    rates_frame = pd.DataFrame(rates)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

    # Encontrar precio de cierre mas cercano a la fecha final
    try:
        price = \
        rates_frame.iloc[np.argmin(abs(rates_frame["time"] - datetime.strptime(str(end_date), "%Y-%m-%d %H:%M:%S")))][
            "close"]
    except:
        price = 0

    return price

# Funcion para obtener el % de sesgos cognitivos
def f_be_de(data: pd.DataFrame, bool_sensibilidad_dec: Optional[bool] = True):
    # Obtener Operaciones con Profit
    with_profit = data[data["Profit"] > 0]

    # Lista temporal
    temp_list = []

    # Iterar cada caso para ver cuales fueron cerrados cuando una operacion con ganancia estaba abierta
    for i in with_profit.index:
        # Seleccionar caso i de profit
        profit_case = with_profit.loc[i]
        # Filtrar casos que estuvieron abiertos al mismo tiempo que caso i
        temp = data[(data["Opentime"] <= profit_case["Closetime"]) & (data["Closetime"] >= profit_case["Closetime"])]
        # Quitar simolos que no traen info
        temp = temp[(temp["Symbol"] != "GOLD") & (temp["Symbol"] != "SILVER")]
        # Agregar ancla como columna
        temp["Ancla"] = profit_case["Closetime"]
        temp["Symbol_Ancla"] = profit_case["Symbol"]
        temp["Type_Ancla"] = profit_case["Type"]
        temp["Volumen_Ancla"] = profit_case["Volume"]
        temp["Precio_del_Ancla"] = profit_case["Closeprice"]
        temp["Profit_Ancla"] = profit_case["Profit"]
        temp["n_Ancla"] = [i] * len(temp)
        temp["num_Ancla"] = list(range(1, len(temp) + 1))
        # Append a lista temporal
        temp_list.append(temp)

    # Hacer data frame
    open_pos = pd.concat(temp_list, ignore_index=True)

    temp_list = []
    # Obtener el precio de x operacion en ancla
    for i in range(len(open_pos)):
        temp_list.append(get_MT5_price(open_pos["Symbol"][i], open_pos["Opentime"][i], open_pos["Ancla"][i]))

    open_pos["Precio_en_Ancla"] = temp_list

    # Definir listas vacias
    dic_ocurrencias = [{"cantidad": len(open_pos)}]
    status_quo = []
    aversion_perdida = []
    sensibilidad = []

    # iterar para encontrar sesgoss en x operaciones
    for index in range(len(open_pos)):
        # Datos del Ancla
        fecha_ancla = open_pos["Ancla"][index]
        precio_ancla = open_pos["Precio_del_Ancla"][index]
        simbolo_ancla = open_pos["Symbol_Ancla"][index]
        direccion_ancla = open_pos["Type_Ancla"][index]
        volumen_ancla = open_pos["Volumen_Ancla"][index]
        profit_ancla = open_pos["Profit_Ancla"][index]
        # Datos de Ocurrencia
        precio_occ_init = open_pos["Openprice"][index]
        precio_occ = open_pos["Precio_en_Ancla"][index]
        simbolo_occ = open_pos["Symbol"][index]
        direccion_occ = open_pos["Type"][index]
        volumen_occ = open_pos["Volume"][index]
        # Encontrar profit, si es buy entonces el de abajo, si no, es al contrario
        profit_occ = (precio_occ - precio_occ_init) * float(volumen_occ) * f_pip_size(simbolo_occ)
        profit_occ = profit_occ if direccion_occ == "Buy" else -profit_occ
        # Profit acumulado en t
        profit_acm = open_pos["profit_acm"][index]
        # Encontrar el ancla
        ancla = open_pos["n_Ancla"][index]
        # contar cantidad
        cantidad = max(open_pos[open_pos["n_Ancla"] == ancla]["num_Ancla"])
        # Llenar diccionario con las caracteristicas de cada ocurrencia
        temp_ocurrencias = {
            "timestamp": fecha_ancla,
            "operaciones": {
                "ganadoras": {
                    "instrumento": simbolo_ancla,
                    "volumen": volumen_ancla,
                    "sentido": direccion_ancla,
                    "profit_ganadora": profit_ancla
                },
                "perdedoras": {
                    "instrumento": simbolo_occ,
                    "volumen": volumen_occ,
                    "sentido": direccion_occ,
                    "profit_perdedora": profit_occ
                }
            },
            "ratio_cp_profit_acm": abs(profit_occ / profit_acm),
            "ratio_cg_profit_acm": abs(profit_ancla / profit_acm),
            "ratio_cp_cg": abs(profit_occ / profit_ancla)
        }

        # Ver que numero de ocurrencia es el ancla
        ocurrencia_ancla = open_pos["num_Ancla"][index]
        # copiar n ocurrencia a diccionario final
        temp_2 = {
            f"ocurrencia_{ocurrencia_ancla}": temp_ocurrencias
        }

        dic_ocurrencias.append(temp_2)

        # Sacar status quo
        sq = abs(profit_occ / profit_acm) < abs(profit_ancla / profit_acm)
        status_quo.append(sq)

        # Sacar aversion perdida
        ap = abs(profit_occ / profit_ancla) > 2
        aversion_perdida.append(ap)

        # Sacar sensibilidad decreciente en %, es decir de n operaciones
        if index > 0 and index < len(with_profit):
            sensibilidad.append(sum([
                with_profit["profit_acm"].values[index] < with_profit["profit_acm"].values[index - 1],
                with_profit["Profit"].values[index - 1] > with_profit["Profit"].values[index] and \
                open_pos["Profit"].values[index - 1] > open_pos["Profit"].values[index],
                open_pos["Profit"].values[index - 1] / with_profit["Profit"].values[index - 1] > 2
            ]) >= 2)

    # Sacar sensibilidad decreciente en caso de que se quiera valor si o no de la ultima operacion
    if bool_sensibilidad_dec:
        sensibilidad = sum([
            with_profit["profit_acm"].values[0] < with_profit["profit_acm"].values[-1],
            with_profit["Profit"].values[-1] > with_profit["Profit"].values[0] and \
            open_pos["Profit"].values[-1] > open_pos["Profit"].values[0],
            open_pos["Profit"].values[-1] / with_profit["Profit"].values[-1] > 2
        ]) >= 2

    # Definir df
    dataframe = pd.DataFrame.from_dict({
        "ocurrencias": [len(open_pos)],
        "status_quo": [f"{sum(status_quo) / len(status_quo) * 100:.2f} %"],
        "aversion_perdida": [f"{sum(aversion_perdida) / len(aversion_perdida) * 100:.2f} %"],
        "sensibilidad_decreciente": ["Si" if sensibilidad else "No"] if bool_sensibilidad_dec \
            else [f"{sum(sensibilidad) / len(sensibilidad) * 100:.2f} %"],
    })

    # Desempacar valores para que pase de lista a diccionario
    ocurrencias_final = {}
    for j in range(len(dic_ocurrencias)):
        ocurrencias_final.update(**dic_ocurrencias[j])

    # Regresar resultado
    return {"ocurrencias": ocurrencias_final, "resultados": {"dataframe": dataframe}}
