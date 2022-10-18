import pandas as pd
import numpy as np
import plotly.graph_objects as go


def graph_pie(stats):
    
    df = stats.get("df_2_ranking")
    df = df.reset_index()
    
    # cambiar % a valor numerico
    df['rank'] = df['rank'] / 100.0
    #identificar maximos para extraer de la grafica mas adelante
    df['pull'] = np.where(df['rank'] == df['rank'].max(),0.2,0)
    
    # array para pull de valores maximos
    p = np.array(df.pull)
    # array labels
    l = np.array(df.symbol)
    # array values
    v = np.array(df['rank'])

    
    fig = go.Figure(data=[go.Pie(labels=l, values=v, pull=np.array(df.pull))])
    # formatting
    fig.update_layout(title='Ranking (operaciones ganadoras)')
 
    return fig.show()


def graph_dddu(evol, estadisticas_mad):

    evol = evol.set_index('timestamp')
    evol = pd.DataFrame(evol['profit_d_acum'])
    evol = evol.sort_index()

    x= (evol.index)
    y = np.array(evol['profit_d_acum'])

    pru = np.array(estadisticas_mad.valor)

    # Valores que necesitamos de la funcion
    #drawdown fecha inicial
    fidd = pru[2]
    #drawdown fecha final
    ffdd = pru[3]
    #drawup fecha inicial
    fidu = pru[5]
    #drawup fecha final
    ffdu = pru[6]

    # DF de DrowUp
    dfdu1 = pd.DataFrame(evol.loc[fidu]).T
    dfdu2 = pd.DataFrame(evol.loc[ffdu]).T
    dfdu = dfdu1.append(dfdu2)

    # DF de DrowDown
    dfdd1 = pd.DataFrame(evol.loc[fidd]).T
    dfdd2 = pd.DataFrame(evol.loc[ffdd]).T
    dfdd = dfdd1.append(dfdd2)


    # Datos DU para graficar
    xu = (dfdu.index)
    yu = np.array(dfdu['profit_d_acum'])

    # Datos DD para graficar
    xd = (dfdd.index)
    yd = np.array(dfdd['profit_d_acum'])

    fig = go.Figure()

    # Evolucion del capital
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        name = 'Evolución del capital',
        marker=dict( color='black',size=10),
        line=dict(color='black', width=4)
    ))
    # DrowDown
    fig.add_trace(go.Scatter(
        x=xd,
        y=yd,
        name = 'DrawDown',
        #marker=dict( color='black',size=10),
        line=dict(color='red', width=4, dash='dash')
    ))

    # DrowUp
    fig.add_trace(go.Scatter(
        x=xu,
        y=yu,
        name = 'DrawUp',
        #marker=dict( color='black',size=10),
        line=dict(color='green', width=4, dash='dash')
    ))


    fig.update_layout(title='DrawDown y DrawUp',
                       xaxis_title='Fecha',
                       yaxis_title='Capital')

    return fig.show()

def graph_bars(dispositoneffect):

    ocurrencias = dispositoneffect.ocurrencias[0]

    df = dispositoneffect[['status_quo','aversion_perdida', 'sensibilidad_decreciente']]
    
    #nombres columnas
    a = list(df.columns)
    
    # valores
    sq = df['status_quo'].apply(lambda x: float("".join([i for i in x if i.isdigit()]))/10000)
    ap = df['aversion_perdida'].apply(lambda x: float("".join([i for i in x if i.isdigit()]))/10000)
    so = df['sensibilidad_decreciente'].apply(lambda x: float("".join([i for i in x if i.isdigit()]))/10000)

    fig = go.Figure([go.Bar(x=a, y=[sq[0]*ocurrencias, ap[0]*ocurrencias, so[0]*ocurrencias])])
    fig.update_layout(title='Disposition Effect',
                        xaxis_title='Disposition Effect')

    return fig.show()