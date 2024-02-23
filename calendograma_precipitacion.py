# # -*- coding: utf-8 -*-
# """
# Created on Fri Jul 14 13:12:32 2023

# @author: easalazarm
# """

import pandas as pd
import calplot
from plotly_calplot import calplot
import datetime
from datetime import date, timedelta
import july
from july.utils import date_range
import matplotlib.colors
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import calmap
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.image as image
import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import dash
from dash import Dash, html, dcc
import plotly.express as px
import math
from dash import Dash, dcc, html, Input, Output, callback
import locale





base = date.today()
ayer = base - timedelta(days = 1)
ayer_str = int(ayer.strftime("%d"))
hoy = int(base.strftime("%d"))
date_list = [ayer - timedelta(days= x) for x in reversed(range(ayer_str))]




seg_estaciones = pd.read_excel(r"O:\Mi unidad\OSPA\02. Datos Diarios\Tablas\Datos hidrometeorologicos\Seguimientos\SEGUIMIENTO_PREC_2023.xlsm", sheet_name = 'PREC_2023')
seg_estaciones = seg_estaciones.loc[seg_estaciones['TIPO'] == 'CON']
estaciones = seg_estaciones["CODIGO"].tolist()



# In[27]:


dep_estaciones = seg_estaciones['DEPARTAMENTO'].tolist()
nom_estaciones = seg_estaciones['ESTACION'].tolist()


# In[36]:


options_dict = dict(zip(nom_estaciones, estaciones))
departments_dict = dict(zip(estaciones, dep_estaciones))




# In[68]:


app = Dash(__name__)




app.layout = html.Div([
      html.H1("Precipitación diaria estaciones convencionales", style={'textAlign': 'center', "font-family": "arial narrow", "color": "#635F5F"}),
    html.H5("Información con datos preliminares de la red de alertas", style={'textAlign': 'center', "font-family": "arial narrow", "color": "#635F5F"}),
        dcc.Dropdown(options = [{'label':f"{word} - {tag} - {departments_dict[tag]} ",'value':tag} for word, tag in options_dict.items()], value = 47060010, id='demo-estaciones', style={"font-family": "arial narrow"} ),
    html.Br(),
    dcc.Graph(
        id='example-graph'
    ),
    html.H6("Nota:Casillas en rojo muestran valores por encima de los 50 mm",
            style={'textAlign': 'left', "font-family": "arial narrow", "color": "#635F5F"})
])

@app.callback(
    Output('example-graph', "figure"),
    #Output('dd-output-line', "figure"),
    Input('demo-estaciones', 'value')
)             
 

def update_grafica(value):
    
    #seg_temporal = seg_estaciones.loc[seg_estaciones['TIPO'] == 'CON']
    seg_temporal = seg_estaciones.loc[seg_estaciones['CODIGO'] == value, :]
    
    columnas_quitar = ['CODIGO', 'LONGITUD', 'LATITUD', 'ELEVACION','ESTACION','MUNICIPIO', 'DEPARTAMENTO', 'R_01',
    'R_02', 'R_03', 'R_04', 'R_05', 'R_06', 'R_07', 'R_08', 'R_09', 'R_10', 'R_11', 'R_12', 'R_T', 'TT_01', 'TT_02', 'TT_03', 'TT_04',
    'TT_05', 'TT_06', 'TT_07', 'TT_08', 'TT_09', 'TT_10', 'TT_11', 'TT_12', 'MM_01', 'MM_02', 'MM_03', 'MM_04', 'MM_05', 'MM_06',
    'MM_07', 'MM_08', 'MM_09', 'MM_10', 'MM_11', 'MM_12', 'AN_01', 'AN_02', 'AN_03', 'AN_04', 'AN_05', 'AN_06', 'AN_07',
    'AN_08', 'AN_09', 'AN_10', 'AN_11', 'AN_12', 'MH_01', 'MH_02', 'MH_03', 'MH_04', 'MH_05', 'MH_06', 'MH_07', 'MH_08',
    'MH_09', 'MH_10', 'MH_11', 'MH_12', 'MX_01', 'MX_02', 'MX_03', 'MX_04', 'MX_05', 'MX_06', 'MX_07','MX_08',
    'MX_09', 'MX_10','MX_11', 'MX_12', '1D_01','2D_01', '3D_01', '1D_02','2D_02', '3D_02', '1D_03', '2D_03', '3D_03',
    '1D_04', '2D_04', '3D_04', '1D_05', '2D_05', '3D_05', '1D_06', '2D_06', '3D_06', '1D_07','2D_07','3D_07',
    '1D_08', '2D_08', '3D_08', '1D_09', '2D_09', '3D_09', '1D_10', '2D_10', '3D_10','1D_11','2D_11','3D_11','1D_12',
    '2D_12', '3D_12', 'R_01*', 'R_02*', 'R_03*', 'R_04*', 'R_05*', 'R_06*', 'R_07*', 'R_08*', 'R_09*','R_10*',
    'R_11*', 'R_12*', 'AN_01*', 'AN_02*','AN_03*', 'AN_04*', 'AN_05*', 'AN_06*', 'AN_07*', 'AN_08*', 'AN_09*', 'AN_10*',
    'AN_11*', 'AN_12*','TIPO', 'REG', 'AH', 'ZH','SZH']
    
    seg_prec = seg_temporal
    seg_prec = seg_prec.loc[:, ~seg_prec.columns.isin(columnas_quitar)]
    ayer_str_calmap = ayer.strftime('%Y-%m-%d %H:%M:%S')
    seg_prec.columns = seg_prec.columns.astype(str)
    
    seg_prec = seg_prec.loc[:, :ayer_str_calmap]
    seg_prec.dropna(axis = 0, how = "all", inplace = True)
    seg_prec = seg_prec.apply(pd.to_numeric, errors = 'coerce')
    
    seg_prec_2 = seg_prec
    seg_prec_2.loc['Total', :] = seg_prec_2.mean(axis=0)
    
    seg_prec_3 = seg_prec_2.drop(seg_prec_2.index.to_list()[:-1] ,axis = 0 )
    #seg_prec_3.to_excel(f"{departamento}-prec.xlsx")
    seg_prec_3 = seg_prec_3.T
    seg_prec_3.index = pd.to_datetime(seg_prec_3.index)
    
    seg_prec_4 = pd.Series(seg_prec_3["Total"].tolist(), index = seg_prec_3.index)
    
    
    
    df = seg_prec_4.reset_index().rename(columns={'index': 'Índice', 0: 'Valor'})
    
    
    fig = calplot(
        df,
        x="Índice",
        y="Valor", 
        gap = 3.5,
        colorscale=[ # Define tu escala de colores personalizada
        [0, '#d2d2d2'], # si es 0, color blanco
        [1/360, '#269199'], # si es mayor que 0, azul
        [40/360, '#269199'], # y menor que 40, azul
        [50/360, '#c75668'], # si es mayor de 50, rojo
        [1, '#c75668']  # si es mayor de 50, rojo
    ],
        #colorscale = [(0.0, "#e9f2f9"), (0.01, "#9cc4e4"), (0.25, "#3a89c9"), (0.50, "#1b325f"), (0.75,"#f26c4f"), (1,"#f26c4f")],
        title = "Precipitación diaria estaciones convencionales (mm)",
        years_title= True,
        name = "Precipitación",
        month_lines_color = "black",
        month_lines_width = 1.6,
        cmap_min = 0,
        cmap_max = 360
    )
    
    
    
    
    return fig



if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




