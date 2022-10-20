from data import data_Nicolas, data_Santiago, data_Ivan, data_Carolina
from functions import f_estadisticas_ba, f_evolucion_capital, f_estadisticas_mad, f_be_de
import visualizations as vn

# Analisis Inicial de Datos
data_Nicolas.head(3)
data_Santiago.head(3)
data_Ivan.head(3)
data_Carolina.head(3)

# Estadistica Descriptiva
ba_Nicolas = f_estadisticas_ba(data_Nicolas)
ba_Santiago = f_estadisticas_ba(data_Santiago)
ba_Ivan = f_estadisticas_ba(data_Ivan)
ba_Carolina = f_estadisticas_ba(data_Carolina)

# Medidas de Atribucion al Desempeño
evolucion_Nicolas = f_evolucion_capital(data_Nicolas)
mad_Nicolas = f_estadisticas_mad(evolucion_Nicolas)

evolucion_Santiago = f_evolucion_capital(data_Santiago)
mad_Santiago = f_estadisticas_mad(evolucion_Santiago)

evolucion_Ivan = f_evolucion_capital(data_Ivan)
mad_Ivan = f_estadisticas_mad(evolucion_Ivan)

evolucion_Carolina = f_evolucion_capital(data_Carolina)
mad_Carolina = f_estadisticas_mad(evolucion_Carolina)

# Behavioral Finance
BF_Nicolas = f_be_de(data_Nicolas)
BF_Santiago = f_be_de(data_Santiago)
BF_Ivan = f_be_de(data_Ivan)
BF_Carolina = f_be_de(data_Carolina)

# Visualizaciones

# Nicolas
# Grafica 1
vn.graph_pie(ba_Nicolas)
# Grafica 2
vn.graph_dddu(evolucion_Nicolas, mad_Nicolas)
# Grafica 3
vn.graph_bars(f_be_de(data_Nicolas, False))

# Santiago
# Grafica 1
vn.graph_pie(ba_Santiago)
# Grafica 2
vn.graph_dddu(evolucion_Santiago, mad_Santiago)
# Grafica 3
vn.graph_bars(f_be_de(data_Santiago, False))

# Ivan
# Grafica 1
vn.graph_pie(ba_Ivan)
# Grafica 2
vn.graph_dddu(evolucion_Ivan, mad_Ivan)
# Grafica 3
vn.graph_bars(f_be_de(data_Ivan, False))

# Carolina
# Grafica 1
vn.graph_pie(ba_Carolina)
# Grafica 2
vn.graph_dddu(evolucion_Carolina, mad_Carolina)
# Grafica 3
vn.graph_bars(f_be_de(data_Carolina, False))

