from functions import f_leer_archivo

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
