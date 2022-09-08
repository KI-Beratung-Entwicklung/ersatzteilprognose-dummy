# Core Pkgs
from turtle import width
import streamlit as st
import streamlit.components.v1 as stc 
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
#from module import *
#from crud_operations import *
from math import ceil
import altair as alt
import pymysql
from PIL import Image
import io
from io import StringIO, BytesIO
img = Image.open("Original on Transparent.png")

# Load EDA Pkg
import pandas as pd 
import numpy as np 

#function to get longitude and latitude data from country name
import pycountry 

# Load Data Vis Pkg
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

import base64
import time
from datetime import datetime
timestr = time.strftime("%Y%m%d-%H%M%S")
import pickle

#global varianles
modelname = 'gbr_model.pkl'
modelname_azure = 'model.pkl'
modelname_nn = 'gridnn_model.pkl'

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
@st.experimental_memo(suppress_st_warning=True)
def aggrid_interactive_table(df: pd.DataFrame):
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )
    return selection

@st.cache
def getData (query):
        conn = pymysql.connect(database = 'data', user = 'root', password = '')
        cursor = conn.cursor()
        df = pd.read_sql(query, conn)
        return df

@st.cache
def getDf(path):
        df = pd.read_csv(path)
        return df


label_dict = {"No":0,"Yes":1}
gender_map = {"Female":0,"Male":1}
target_label_map = {"Negative":0,"Positive":1}

html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Early Stage DM Risk Data App </h1>
		<h4 style="color:white;text-align:center;">Diabetes </h4>
		</div>
		"""

def search(regex: str, df, case=False):
    """Search all the text columns of `df`, return rows with any matches."""
    textlikes = df.select_dtypes(include=[object, "string"])
    return df[
        textlikes.apply(
            lambda column: column.str.contains(regex, regex=True, case=case, na=False)
        ).any(axis=1)
    ]

def make_downloadable_df_format(data, format_type="csv"):
        if format_type == "json":
                datafile = data.to_json()
        else:
                datafile = data.to_csv(index=False)

        b64 = base64.b64encode(datafile.encode()).decode()  # B64 encoding
        st.markdown("###  Download File  ### ")
        new_filename = "ersatzteil_download_{}.{}".format(timestr, format_type)
        href = f'<a href="data:file/{format_type};base64,{b64}" download="{new_filename}">Click Here!</a>'
        st.markdown(href, unsafe_allow_html=True)

def get_fvalue(val):
	feature_dict = {"No":0,"Yes":1}
	for key,value in feature_dict.items():
		if val == key:
			return value 

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def main():
        st.set_page_config(page_title="Ersatzteilübersicht", page_icon='🏠', layout="wide",
                initial_sidebar_state="auto")
        #df = getData("SELECT * FROM wareneingang Order by bestell_nr desc LIMIT 100")
        df = getDf('df_fuer_referenzprojekt.csv').tail(188347)
        land = getDf('maschinendaten_auf_karte.csv')
        df.drop(df.columns[[0,1,2]],axis=1,inplace=True)
        df["Artikelnummer"] = df["Artikelnummer"].astype('category')
        df_ready_for_ml = getDf('df_ml_ready_fuer_referenzprojekt.csv')
        maschinennummern=[3026780, 3037371, 3022760, 3043200, 3036360, 3011740, 3042120,
       3036120, 3035130, 3040250, 3038580, 3045050, 3042460, 3037260,
       3016631, 3034170, 3043760, 3037805, 3043070, 3034770, 3044070,
       3042851, 3042530, 3040244, 3037502, 3036660, 3030290, 3031190,
       3021300, 3038910, 3036170, 3028772, 3038780, 3042201, 3035370,
       3026660, 3042700, 3028940, 3033740, 3044432, 3037850, 3035740,
       3031580, 3039670, 3038821, 3025901, 3032200, 3021291, 3042940,
       3043420, 3021001, 3035550, 3400150, 3031180, 3040230, 3036110,
       3035570, 3044230, 3025020, 3042210, 3025150, 3037030, 3032650,
       3023011, 3037520, 3034150, 3038030, 3039170, 3019092, 3032850,
       3027500, 3028042, 3043640, 3037421, 3032830, 3044810, 3030480,
       3038430, 3021010, 3041762, 3019800, 3039690, 3013770, 3035690,
       3038770, 3022851, 3041251, 3011720, 3040192, 3027940, 3036150,
       3028620, 3042160, 3041400, 3035972, 3031120, 3034920, 3037471,
       3040110, 3040482, 3040641, 3044500, 3040810, 3023500, 3023372,
       3037980, 3041261, 3041530, 3044550, 3039110, 3042191, 3014261,
       3032860, 3036130, 3041900, 3026030, 3038570, 3029800, 3028282,
       ]
        maschinentypK03100 = df[df['Maschinennummer'].isin(maschinennummern)]
        st.sidebar.image(img)
        rad = st.sidebar.selectbox("Menü", ["Übersicht","Datenvisualisierung","Data Prediction","Kontakt"])

        if rad == "Übersicht":

                st.title("Übersicht")

                st.subheader('Karte nach Maschinenstandorten')
                
                choice0 = st.multiselect("Wählen Sie die Maschinen:",np.sort(df['Maschinennummer'].unique()), key = 0)
                if (len(choice0)!=0):
                        dff0 = land[land['Maschinennummer'].isin(choice0)]
                        #empty map
                        world_map= folium.Map(tiles="cartodbpositron")
                        marker_cluster = MarkerCluster().add_to(world_map)
                        #for each coordinate, create circlemarker of user percent
                        for i in range(len(dff0)):
                                lat = dff0.iloc[i]['latitude']
                                long = dff0.iloc[i]['longitude']
                                radius=5
                                popup_text = """Country : {}<br><br>
                                        Maschinennummer : {}<br><br>
                                        Bisheriges Servicevolumen : {} EUR<br><br>
                                        Verkaufte Aufträge : {}<br><br>
                                        Verkaufte Menge : {}<br>"""
                                popup_text = popup_text.format(dff0.iloc[i]['name'],
                                                        dff0.iloc[i]['Maschinennummer'],
                                                        dff0.iloc[i]['Nettowert'],
                                                        dff0.iloc[i]['Verkaufsbeleg'],
                                                        dff0.iloc[i]['Menge']
                                                        )
                                folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
                        folium_static(world_map, width=1000)

                else:
                        #empty map
                        world_map1= folium.Map(tiles="cartodbpositron")
                        marker_cluster1 = MarkerCluster().add_to(world_map1)
                        #for each coordinate, create circlemarker of user percent
                        for i in range(len(land)):
                                lat = land.iloc[i]['latitude']
                                long = land.iloc[i]['longitude']
                                radius=5
                                popup_text = """Country : {}<br><br>
                                        Maschinennummer : {}<br><br>
                                        Bisheriges Servicevolumen : {} EUR<br><br>
                                        Verkaufte Aufträge : {}<br><br>
                                        Verkaufte Menge : {}<br>"""
                                popup_text = popup_text.format(land.iloc[i]['name'],
                                                        land.iloc[i]['Maschinennummer'],
                                                        land.iloc[i]['Nettowert'],
                                                        land.iloc[i]['Verkaufsbeleg'],
                                                        land.iloc[i]['Menge']
                                                        )
                                folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster1)
                        folium_static(world_map1, width=1000)

                st.header("")

                col1, col2, col3 = st.columns(3)
                col1.metric(label="Ersatzteilauftragspositionen insgesamt", value = len(df), delta="+280 gestern")
                col2.metric(label="Letzte Lieferung", value = df['Auftragsdatum'].max())
                col3.metric(label="Material insgesamt", value = len(df['Artikelnummer'].unique()), delta="1 %")

                st.header("")

                st.subheader('Nettowert aller Ersatzteilaufträge pro Land')
                path_to_html = "nettowert_per_country_map.html" 
                 # Read file and keep in variable
                with open(path_to_html,'r') as f: 
                        map = f.read()
                # Show in webpage
                st.components.v1.html(map,height=400)

                st.header("")

                st.subheader("Auflistung aller Ersatzteilaufträge")
                selection = aggrid_interactive_table(df=df)


        if rad == "Datenvisualisierung":

                ansicht = st.sidebar.radio("Wählen Sie die Ansicht:",['Alle Maschinen','Ausgewählte Maschinen'])

                if ansicht == "Alle Maschinen":    

                        st.title("Datenvisualisierung")

                        st.write('Nettowert aller Ersatzteilaufträge pro Land:')
                        path_to_html = "nettowert_per_country_map_dots.html" 
                        # Read file and keep in variable
                        with open(path_to_html,'r') as f: 
                                map_dots = f.read()
                        ## Show in webpage
                        st.components.v1.html(map_dots,height=400)


                        with st.expander("Ersatzteilbedarfsverlaufkurve aller Maschinen"):
                                choice = st.radio("Wählen Sie die Datengrundlage:",['Artikelgruppenbasis','Artikelbasis'])
                                if choice == "Artikelgruppenbasis":
                                        path_to_htmll = "Ersatzteil-Bedarfsverlauf_alle_Maschinen.html"
                                else:
                                        path_to_htmll = "Ersatzteil-Bedarfsverlauf_alle_Maschinen_Artikelebene.html"                
                                # Read file and keep in variable
                                with open(path_to_htmll,'r') as f: 
                                        map_dots = f.read()
                                ## Show in webpage
                                st.components.v1.html(map_dots,height=400)
                                st.write("* Dots sind die jeweiligen Ersatzteilaufträge. Größe der Dots enttspricht der Auftragsmenge.")
                                st.write("")

                        with st.expander("Verlaufskurven ausgewählter Werte aller Maschinen"):
                                col1, col2, col3 = st.columns(3)
                                col1.metric(label="Nettowert insgesamt in TEUR", value = ((df['Nettowert'].sum())/1000).round(1), delta="+14 TEUR gestern")
                                col2.metric(label="Verkaufte Menge ingesamt", value = df['Menge'].sum())
                                col3.metric(label="Ersatzteilaufträge insgesamt", value = len(df), delta="+421 Aufträge diese Woche")
                                st.write("Fakturierte Menge, Nettowert und Anzahl der Positionen im Zeitverlauf")
                                path_to_htmlll='Fakturierte Menge, Nettowert und Anzahl der Positionen im Zeitverlauf.html'
                                with open(path_to_htmlll,'r') as f: 
                                        maps = f.read()
                                ## Show in webpage
                                st.components.v1.html(maps,height=400)

                        with st.expander('Verteilung der Artikelgruppen aller Maschinen'):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                        number = st.slider("Wie viele Artikelgruppen sollen angezeigt werden",1,50)
                                mat_gp = df[['Artikelgruppe','Menge']]
                                mat_gp = mat_gp.groupby('Artikelgruppe').sum()
                                mat_gp.reset_index(inplace = True)
                                mat_gp = mat_gp.sort_values('Menge', ascending = False)
                                #the top 20
                                mat_gp2 = mat_gp[:number].copy()                                
                                #others
                                new_row = pd.DataFrame(data = {
                                'Artikelgruppe' : ['Sonstige'],
                                'Menge' : [mat_gp['Menge'][number:].count()]
                                })                                
                                #combining top 20 with others
                                mat_gp2 = pd.concat([mat_gp2, new_row])
                                #Verteilung Material
                                col1, col2, col3 = st.columns(3)
                                col1.metric(label="Anzahl unterschiedlicher Artikelgruppen", value = len(df['Artikelgruppe'].unique()))
                                col2.metric(label="Häufigste Artikelgruppe", value = 'Kühlungen', delta='+3 Aufträge letzte Woche')
                                col3.metric(label="Anteil der Top5 Artikelgruppen", value = str((mat_gp2[0:5]["Menge"].sum()/(mat_gp2["Menge"].sum())*100).round(2))+" %", delta='- 5.4% in der letzten Woche')    
                                st.write('Verteilung der Artikelgruppen jeder Auftragsposition (auf Basis der Menge)')
                                fig = px.pie(mat_gp2, values='Menge', names='Artikelgruppe', color_discrete_sequence=px.colors.sequential.OrRd)
                                st.plotly_chart(fig)

                        with st.expander("Ersatzteil-Bedarfsverlauf aller Maschinen je Artikelgruppe"):
                                path_to_htmllll='Ersatzteil-Bedarfsverlauf_alle_Maschinen_Artikelgruppe_stackedbar.html'
                                with open(path_to_htmllll,'r') as f: 
                                        maps = f.read()
                                ## Show in webpage
                                st.components.v1.html(maps,height=400)

                else:
                        st.title("Datenvisualisierung")
                        st.write('Nettowert aller Ersatzteilaufträge pro Land:')
                        path_to_html = "nettowert_per_country_map_dots.html" 
                        # Read file and keep in variable
                        with open(path_to_html,'r') as f: 
                                map_dots = f.read()
                        ## Show in webpage
                        st.components.v1.html(map_dots,height=400)

                        with st.expander("Ersatzteilbedarfsverlaufkurve ausgewählter Maschinen"):
                                choice1 = st.multiselect("Wählen Sie die Maschinen:",np.sort(df['Maschinennummer'].unique()))
                                dff = df[df['Maschinennummer'].isin(choice1)]
                                col1, col2, col3 = st.columns(3)
                                col1.metric(label="Nettovolumen insgesamt in TEUR", value = ((dff['Nettowert'].sum())/1000).round(1))
                                col2.metric(label="Verkaufte Menge ingesamt", value = dff['Menge'].sum())
                                col3.metric(label="Ersatzteilaufträge insgesamt", value = len(dff))                                
                                choice1 = st.radio("Wählen Sie die Datengrundlage:",['Artikelgruppenbasis','Artikelbasis'])
                                if choice1 == "Artikelgruppenbasis":
                                        if (len(choice1)!=0):                                               
                                                fig = px.scatter(dff, x="Alter der Maschine bei Auftrag", y="Nettowert", color="Artikelgruppe",
                                                                size='Menge', hover_data=['Artikelgruppe','Nettowert','Menge'],color_discrete_sequence=px.colors.sequential.OrRd)
                                                st.plotly_chart(fig)
                                                buffer = io.StringIO()
                                                fig.write_html(buffer, include_plotlyjs='cdn')
                                                html_bytes = buffer.getvalue().encode()

                                                st.download_button(
                                                        label='Download hmtl-Grafik',
                                                        data=html_bytes,
                                                        file_name='downloaded_grafic.html',
                                                        mime='text/html'
                                                )
        
                                else:
                                        if (len(choice1)!=0):
                                                fig1 = px.scatter(dff, x="Alter der Maschine bei Auftrag", y="Nettowert", color="Artikelnummer",
                                                                size='Menge', hover_data=['Artikelgruppe','Nettowert','Menge'],color_discrete_sequence=px.colors.sequential.OrRd)
                                                st.plotly_chart(fig1)
                                
                        with st.expander("Verlaufskurven ausgewählter Werte ausgewählter Maschinen"):
                                choice2 = st.multiselect("Wählen Sie die Maschinen:",np.sort(df['Maschinennummer'].unique()), key = 2)
                                if (len(choice2)!=0):
                                        dff2 = df[df['Maschinennummer'].isin(choice2)]
                                        col1, col2, col3, col4 = st.columns(4)
                                        col1.metric(label="Nettovolumen insgesamt in TEUR", value = ((dff2['Nettowert'].sum())/1000).round(1))
                                        col2.metric(label="Verkaufte Menge ingesamt", value = dff2['Menge'].sum())
                                        col3.metric(label="Ersatzteilaufträge insgesamt", value = len(dff2))
                                        col4.metric(label="Durchschnittliches Alter der Maschinen",value = (str(round(dff2['Alter der Maschine bei Auftrag'].mean(), 2))+ " Jahre")) 
                                        st.write("Fakturierte Menge, Nettowert und Anzahl der Positionen im Zeitverlauf")
                                        
                                        data = dff2.groupby('Alter der Maschine bei Auftrag').agg({'Verkaufsbeleg':'count','Nettowert':'sum','Menge':'sum','Artikelgruppe':'first'})
                                        data.rename(columns={'Menge':'Kumulierte Menge','Verkaufsbeleg':'Anzahl der Positionen'}, inplace = True)
                                        data = data.reset_index()
                                        fig2 = px.line(data, x="Alter der Maschine bei Auftrag", y=['Kumulierte Menge','Nettowert','Anzahl der Positionen'],
                                                color_discrete_sequence=['#fee8c8', '#fdbb84','#bf4a33'])
                                        st.plotly_chart(fig2)
                                
                        with st.expander('Verteilung der Artikelgruppen ausgewählter Maschinen'):
                                choice3 = st.multiselect("Wählen Sie die Maschinen:",np.sort(df['Maschinennummer'].unique()), key = 3)
                                if (len(choice3)!=0):
                                        dff3 = df[df['Maschinennummer'].isin(choice3)]
                                        mat_gp = dff3[['Artikelgruppe','Menge']]
                                        mat_gp = mat_gp.groupby('Artikelgruppe').sum()
                                        mat_gp.reset_index(inplace = True)
                                        mat_gp = mat_gp.sort_values('Menge', ascending = False)
                                        #the top 20
                                        mat_gp2 = mat_gp[:20].copy()                                
                                        #others
                                        new_row = pd.DataFrame(data = {
                                        'Artikelgruppe' : ['Sonstige'],
                                        'Menge' : [mat_gp['Menge'][20:].count()]
                                        })                                
                                        #combining top 20 with others
                                        mat_gp2 = pd.concat([mat_gp2, new_row])
                                        #Verteilung Material
                                        col1, col2, col3 = st.columns(3)
                                        col1.metric(label="Nettovolumen insgesamt in TEUR", value = ((dff3['Nettowert'].sum())/1000).round(1))
                                        col2.metric(label="Verkaufte Menge ingesamt", value = dff3['Menge'].sum())
                                        col3.metric(label="Ersatzteilaufträge insgesamt", value = len(dff3))    
                                        st.write('Verteilung der Artikelgruppen jeder Auftragsposition (auf Basis der Menge)')
                                        fig = px.pie(mat_gp2, values='Menge', names='Artikelgruppe', color_discrete_sequence=px.colors.sequential.OrRd)
                                        st.plotly_chart(fig)

                
                
        if rad == "Data Prediction":

                view = st.sidebar.radio("Wählen Sie die Ansicht:",['Prognose auf Einzelteilbasis','Prognose nach Artikelgruppen'])

                if view == "Prognose auf Einzelteilbasis": 
                
                        st.title("Ersatzteilprognose auf Einzelteilbasis")

                        st.write("Im folgenden Beispiel soll der Ersatzteilbedarf eines einzelnen Artikels auf 180 Tage vorhergesagt werden. Bitte geben Sie dazu eine beliebige Artikelnummer im untenstehenden Eingabefeld ein.")

                        st.info("Natürlich ist der Prognosezeitraum frei und individuell von Ihnen wählbar. Desweiteren kann das Prognosemodell dynamisch die jeweiligen halbjährigen Bedarfe direkt an Ihre Datenbank zurückspielen. Diese Informationen können von Ihnen dann in Ihren eigenen Systemen und Prozessen flexibel genutzt werden. Ein seperates Tool (wie dieses hier) ist dann nicht notwendig. Bei Fragen melden Sie sich unter www.danielbluemlein.de")

                        today = datetime.now()
                        st.write("Datenstand vom:", today)

                        with st.spinner('Daten werden geladen... bitte warten!'):
                                selection = aggrid_interactive_table(df=df)
                                st.header('180 Tage Forecast')
                                st.info("Die im Rahmen des 180 Tage Forecasts ermittelte Zahl gibt an, wie häufig der betrachtete Artikel im nächsten halben Jahr nachgefragt wird. Diese Information gibt dem Einkauf bzw. der eigenen Fertigung eine gewisse Vorlaufzeit, die Ersatzteile zu beschaffen bzw. zu fertigen, sodass diese zum Bedarfszeitpunkt direkt auf Lager sind.")
                                artikelnummer = st.text_input("Artikelnummer eingeben")
                                with open(modelname_nn,'rb') as model_file:
                                                model = pickle.load(model_file)
                        
                                if len(artikelnummer) > 0:
                                        input_data = df_ready_for_ml[df_ready_for_ml['Id']==int(artikelnummer)].tail(1)
                                        if int(artikelnummer) in df['Artikelnummer'].unique():
                                                st.write('Input-Data:')
                                                st.dataframe(input_data)
                                                st.success('Der Artikel '+artikelnummer+' wird in den nächsten 180 Tagen '+str(model.predict(input_data)[0].round(3))+' Mal nachgefragt!')
                                                st.balloons()
                                
                                        else:
                                                st.error('Bitte geben Sie eine gültige Artikelnummer ein!')
                                else:
                                        st.error('Bitte geben Sie eine Artikelnummer ein!')

                                # import urllib.request
                                # import json
                                # import os
                                # import ssl

                                # def allowSelfSignedHttps(allowed):
                                # # bypass the server certificate verification on client side
                                #         if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
                                #                 ssl._create_default_https_context = ssl._create_unverified_context

                                # allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

                                # # Request data goes here
                                # # The example below assumes JSON formatting which may be updated
                                # # depending on the format your endpoint expects.
                                # # More information can be found here:
                                # # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
                        

                                # data ={
                                # "Inputs": {
                                # "data": [
                                # {
                                #         "Column2": "xxxxxx",
                                #         "Id": int(input_data.Id.values[0]),
                                #         "DailyOutput": float(input_data.DailyOutput.values[0]),
                                #         "DailyOutput_lag30": float(input_data.DailyOutput_lag30.values[0]),
                                #         "DailyOutput_lag60": float(input_data.DailyOutput_lag60.values[0]),
                                #         "DailyOutput_lag90": float(input_data.DailyOutput_lag90.values[0]),
                                #         "DailyOutput_lag180": float(input_data.DailyOutput_lag180.values[0]),
                                #         "DailyOutput_lag270": float(input_data.DailyOutput_lag270.values[0]),
                                #         "DailyOutput_lag360": float(input_data.DailyOutput_lag360.values[0]),
                                #         "DailyOutput_Change_30": float(input_data.DailyOutput_Change_30.values[0]),
                                #         "DailyOutput_Change_60": float(input_data.DailyOutput_Change_60.values[0]),
                                #         "DailyOutput_Change_90": float(input_data.DailyOutput_Change_90.values[0]),
                                #         "DailyOutput_Change_180": float(input_data.DailyOutput_Change_180.values[0]),
                                #         "DailyOutput_Change_270": float(input_data.DailyOutput_Change_270.values[0]),
                                #         "DailyOutput_Change_360": float(input_data.DailyOutput_Change_360.values[0]),
                                #         "dayofweek": int(input_data.dayofweek.values[0]),
                                #         "dayofyear": int(input_data.dayofyear.values[0]),
                                #         "sin_day": float(input_data.sin_day.values[0]),
                                #         "cos_day": float(input_data.cos_day.values[0]),
                                #         "dayofmonth": int(input_data.dayofmonth.values[0])
                                #         }
                                # ]
                                # },
                                # "GlobalParameters": 0.0
                                # }

                                # body = str.encode(json.dumps(data))

                                # url = 'http://ce699046-943a-4c8a-acad-9b24791d0449.westeurope.azurecontainer.io/score'
                                # api_key = 'vBe1lH6e53YMsvsvMPK1a6dwCPLVUare' # Replace this with the API key for the web service

                                # # The azureml-model-deployment header will force the request to go to a specific deployment.
                                # # Remove this header to have the request observe the endpoint traffic rules
                                # headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

                                # req = urllib.request.Request(url, body, headers)

                                # try:
                                #         response = urllib.request.urlopen(req)

                                #         result = response.read()
                                #         st.write(type(result))
                                #         st.write(result)
                                #         st.write(result[1])
                                # except urllib.error.HTTPError as error:
                                #         st.write("The request failed with status code: " + str(error.code))

                                #         # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
                                #         st.write(error.info())
                                #         st.write(error.read().decode("utf8", 'ignore'))

                                # #st.write(input_data.DailyOutput_lag30.values[0])


                                #model.predict
                        

                else:

                        st.title("Ersatzteilprognose nach Artikelgruppen")
                        choice4 = st.selectbox("Wählen Sie die Artikelgruppen:",np.sort(df['Artikelgruppe'].unique()), key = 4)
                        if (len(choice4)!=0):
                                with st.spinner('Daten werden geladen... bitte warten!'):
                                        dff4 = df[df['Artikelgruppe']==choice4]
                                        col1, col2, col3, col4 = st.columns(4)
                                        col1.metric(label="Nettovolumen insgesamt in TEUR", value = ((dff4['Nettowert'].sum())/1000).round(1))
                                        col2.metric(label="Verkaufte Menge ingesamt", value = dff4['Menge'].sum())
                                        col3.metric(label="Ersatzteilaufträge insgesamt", value = len(dff4))
                                        col4.metric(label="Durchschnittliches Alter der Maschinen",value = (str(round(dff4['Alter der Maschine bei Auftrag'].mean(), 2))+ " Jahre")) 
                                        fig4 = px.line(dff4, x="Auftragsdatum", y=['Menge','Nettowert'],color_discrete_sequence=['#fee8c8','#bf4a33'],
                                                        title= str("Bedarfsverlauf der Artikelgruppe: " + choice4))
                                        st.plotly_chart(fig4)

                                        buffer = io.StringIO()
                                        fig4.write_html(buffer, include_plotlyjs='cdn')
                                        html_bytes = buffer.getvalue().encode()

                                        st.download_button(
                                                label='Download hmtl-Grafik',
                                                data=html_bytes,
                                                file_name='downloaded_grafic.html',
                                                mime='text/html'
                                                )
                

        if rad == "Kontakt":

                

                cols1, cols2, cols3 = st.columns(3)

                with cols1:
                        st.header("Daniel Blümlein - KI Beratung & Entwicklung")
                        img2 = Image.open("wm_nor.jpg")
                        st.image(img2)
                        st.write("Markgrafenallee 99")
                        st.write("74541 Vellberg")
                        st.write("Tel: 0151 50744422")
                        st.write("Mail: info@danielbluemlein.de")

                with cols3:
                        

                        st.subheader("")
                        st.subheader(":mailbox: Get in Touch With Us")        
                        contact_form = """
                        <form action="https://formsubmit.co/info@danielbluemlein.de" method="POST">
                                <input type="text" name="vorname" placeholder="Vorname" required>
                                <input type="text" name="nachname" placeholder="Nachname" required>
                                <input type="text" name="firma" placeholder="Firma" required>
                                <input type="email" name="email" placeholder="E-Mail" required>
                                <input type="textarea" name="message" placeholder="Was ist Ihr Anliegen?" required>
                                <button type="submit">Absenden</button>
                        </form>
                        """
                        st.markdown(contact_form,unsafe_allow_html=True)

                        def local_css(file_name):
                                with open(file_name) as f:
                                        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True )
                        
                        local_css("style.css")




if __name__ == '__main__':
	main()
