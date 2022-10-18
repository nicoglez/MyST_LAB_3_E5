from functions import f_leer_archivo, f_columnas_tiempos, f_columnas_pips

# Paths de cada integrante
path_Santiago = "files/ReportHistory-5503127_SRR.xlsx"
path_Nicolas = "files/ReportHistory-5500995_SNGV.xlsx"
path_Ivan = "files/ReportHistory-Metatrader_JIL.xlsx"
path_Carolina ="files/ReportHistory_CF.xlsx"

# Datos de cada Integrante
data_Santiago = f_leer_archivo(path_Santiago)
data_Nicolas = f_leer_archivo(path_Nicolas)
data_Ivan = f_leer_archivo(path_Ivan)
data_Carolina = f_leer_archivo(path_Carolina)

# Agregar columnas con funciones creadas

# Agregar columnas de tiempos
data_Santiago = f_columnas_tiempos(data_Santiago)
data_Nicolas = f_columnas_tiempos(data_Nicolas)
data_Ivan = f_columnas_tiempos(data_Ivan)
data_Carolina = f_columnas_tiempos(data_Carolina)

# Agregar columnas de pips
data_Santiago = f_columnas_pips(data_Santiago)
data_Nicolas = f_columnas_pips(data_Nicolas)
data_Ivan = f_columnas_pips(data_Ivan)
data_Carolina = f_columnas_pips(data_Carolina)
