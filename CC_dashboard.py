import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



st.set_page_config(layout="wide")

col1 = st.sidebar
col2, col3 = st.columns((2,1))
col2.title('CC PLAY GROUND')
col2.markdown("""you can play around with experiment-model fit knowledge""")
col2.markdown("""made by  Additive & Green Solution Group at Research and Innovation Center , KaengKhoi , Saraburi""")




data = r'CC.csv'
df = pd.read_csv(data)

df_cut = df.drop(["BET","CC_LOI","Raw_Calcite","Raw_Albite","RUN","CC_Goethite","Raw_SiO2 (%)","Raw_Al2O3 (%)","Raw_Fe2O3 (%)","Raw_CaO (%)","Raw_MgO (%)","Raw_SO3 (%)","Raw_Na2O (%)","Raw_K2O (%)","Raw_TiO2 (%)","Raw_Cr2O3 (%)","Raw_P2O5 (%)","Raw_MnO (%)","Raw_SrO (%)","Raw_ZnO (%)","Raw_ZrO (%)","Raw_Cl (%)","Raw_CuO (%)","Raw_NiO (%)","Raw_Rb2O (%)","Raw_CoO (%)","Raw_V2O5 (%)","Raw_As2O3 (%)","Raw_Y2O3 (%)","CC_SiO2 (%)","CC_Al2O3 (%)","CC_Fe2O3 (%)","CC_CaO (%)","CC_MgO (%)","CC_SO3 (%)","CC_Na2O (%)","CC_K2O (%)","CC_TiO2 (%)","CC_Cr2O3 (%)","CC_P2O5 (%)","CC_MnO (%)","CC_SrO (%)","CC_ZnO (%)","CC_ZrO2 (%)","CC_Cl (%)","CC_CuO (%)","CC_NiO (%)","CC_CoO (%)","CC_Rb2O (%)","CC_Y2O3 (%)","CC_BaO (%)","CC_V2O5 (%)","CC_As2O3 (%)"], axis=1)

########### Make Prediction of 28 Day Strength
X = df_cut[['CC_Quartz',
 'CC_Hematite',
 'CC_Magnetite',
 'CC_Microcline',
 'CC_Anorthite',
 'CC_Anorthoclase',
 'CC_Muscovite',
 'CC_Amorphous',
 'Percent_CK',
'VMD'
]]

y = df_cut[[
 '28DS',
]]


#X['Percent_CK'].astype(np.int64)
#X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, y2, test_size = 0.005, random_state = 40)

rdf = RandomForestRegressor(n_estimators = 100, random_state = 0)
rdf = rdf.fit(X, y)

########### Specify Input Parameters
col1.header('Specify Input Parameters')

def user_input_features():
        
    Percent_CK = col1.slider('Percent_CK', min_value =13 , max_value = 70, value = 35, step= 1)
    CC_Muscovite = col1.slider('CC_Muscovite',min_value =0 , max_value = 20, value = 4, step= 1)
    VMD = col1.slider('VMD', min_value =5 , max_value = 20, value = 16, step= 1)
    CC_Quartz = col1.slider('CC Quartz', min_value =1 , max_value = 55, value = 17, step= 1)
    CC_Anorthite = col1.slider('CC_Anorthite', min_value =0 , max_value = 13, value = 2, step= 1)
    CC_Anorthoclase = col1.slider('CC_Anorthoclase', min_value =0 , max_value = 4, value = 1, step= 1)
    CC_Hematite = col1.slider('CC_Hematite', min_value =0 , max_value = 8, value = 4, step= 1)
    CC_Magnetite = col1.slider('CC_Magnetite', min_value =0.00 , max_value = 1.40, value = 0.50, step= 0.10)
    CC_Microcline = col1.slider('CC_Microcline',min_value =0.0 , max_value = 5.0, value = 1.4, step= 0.2)
    
    
    CC_Amorphous = col1.slider('CC_Amorphous', min_value =40 , max_value = 90, value = 66, step= 2)
    
    data = {'CC_Quartz': CC_Quartz,
            'CC_Hematite': CC_Hematite,
            'CC_Magnetite': CC_Magnetite,
            'CC_Microcline': CC_Microcline,
            'CC_Anorthite': CC_Anorthite,
            'CC_Anorthoclase': CC_Anorthoclase,
            'CC_Muscovite': CC_Muscovite,
            'CC_Amorphous': CC_Amorphous,
            'Percent_CK': Percent_CK,
            'VMD': VMD}
        
       

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()



col2.markdown('Specified Input parameters')

fig = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
                 cells=dict(values=[df.CC_Quartz.map('{:,.2f}'.format), 
                                    df.CC_Hematite.map('{:,.2f}'.format), 
                                    df.CC_Magnetite.map('{:,.2f}'.format),
                                    df.CC_Microcline.map('{:,.2f}'.format), 
                                    df.CC_Anorthite.map('{:,.2f}'.format), 
                                        df.CC_Anorthoclase.map('{:,.2f}'.format),
                                    df.CC_Muscovite.map('{:,.2f}'.format),
                                    df.CC_Amorphous.map('{:,.2f}'.format),
                                    df.Percent_CK.map('{:,.2f}'.format),
                                    df.VMD.map('{:,.2f}'.format)],fill_color='lavender',align='center'))
                ])
fig.update_layout(height=60, width = 600, margin=dict(r=0, l=0, t=0, b=0))
col2.write(fig)

col6, col7 = st.columns((1,1))


prediction28D = rdf.predict(df)

col6.subheader('Prediction of 28 Day Strength = {} ksc'.format(prediction28D))
col6.write('---')

#make prediction at 1DS
X2 = df_cut[['CC_Quartz',
 'CC_Hematite',
 'CC_Magnetite',
 'CC_Microcline',
 'CC_Anorthite',
 'CC_Anorthoclase',
 'CC_Muscovite',
 'CC_Amorphous',
 'Percent_CK',
'VMD'
]]

y2 = df_cut[[
'1DS',
]]
rdf1D = RandomForestRegressor(n_estimators = 100, random_state = 0)
rdf1D = rdf1D.fit(X2, y2)
prediction1D = rdf1D.predict(df)
col6.subheader('Prediction of 1 Day Strength = {} ksc'.format(prediction1D))
col6.write('---')

#make prediction at w/b
X3 = df_cut[['CC_Quartz',
 'CC_Hematite',
 'CC_Magnetite',
 'CC_Microcline',
 'CC_Anorthite',
 'CC_Anorthoclase',
 'CC_Muscovite',
 'CC_Amorphous',
 'Percent_CK',
'VMD'
]]

y3 = df_cut[[
'w/b',
]]
rdfwb = RandomForestRegressor(n_estimators = 100, random_state = 0)
rdfwb = rdfwb.fit(X3, y3)
predictionwb = rdfwb.predict(df)
col6.subheader('Prediction of w/b = {} '.format(predictionwb))
col6.write('---')

col4, col5 = st.columns((1,2))

##############################################28DS surface

CC_Quartz = np.full((100,1),df['CC_Quartz'].iloc[0])                   
CC_Hematite = np.full((100,1),df['CC_Hematite'].iloc[0])      
CC_Magnetite = np.full((100,1),df['CC_Magnetite'].iloc[0] )     
CC_Microcline = np.full((100,1),df['CC_Microcline'].iloc[0])      
CC_Anorthite = np.full((100,1),df['CC_Anorthite'].iloc[0]  )    
CC_Anorthoclase = np.full((100,1),df['CC_Anorthoclase'].iloc[0] )    
#CC_Muscovite = np.full((100,1),df['CC_Muscovite'].iloc[0] )     
CC_Amorphous = np.full((100,1),df['CC_Amorphous'].iloc[0])      
#Percent_CK = np.full((100,1),df['Percent_CK'].iloc[0] )     
VMD =  np.full((100,1),df['VMD'].iloc[0]  )    
                           
                    
min_temp = 0
max_temp = 25
min_pressure = 10
max_pressure = 70
Percent_CK, CC_Muscovite = np.meshgrid(np.linspace(min_temp, max_pressure, 10), np.linspace(min_pressure, max_temp, 10))
predict_x = np.concatenate((CC_Muscovite.reshape(-1, 1), 
                           Percent_CK.reshape(-1, 1)), 
                           axis=1)
CC_Muscovite0 = CC_Muscovite.reshape(-1,1)
Percent_CK0 = Percent_CK.reshape(-1,1)
               
Surface_matrix = np.concatenate((CC_Quartz, CC_Hematite, CC_Magnetite, CC_Microcline,CC_Anorthite, CC_Anorthoclase, CC_Muscovite0, CC_Amorphous,Percent_CK0, VMD),  axis=1)
predict_28DS = rdf.predict(Surface_matrix)
               
x_ = CC_Muscovite
y_ = Percent_CK
z_=  predict_28DS.reshape(Percent_CK.shape)

#z= predict_28DS.reshape(predict_pressure.shape)

SR28 = go.Figure(data=[go.Surface(x=x_, y=y_, z=z_, colorscale='Viridis',showscale=False)])
#SR28.update_layout( autosize=False,
                 #width=250, height=250,
                 #)
SR28.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=-2.5, z=1.25)
)

SR28.update_layout(scene_camera=camera)
SR28.update_layout(scene = dict(
                   xaxis_title='CC_Muscovite',
                   yaxis_title='Percent_CK',
                   zaxis_title='predict_28DS'),
                   width=400,height=350,
                   margin=dict(r=5, b=0, l=0, t=0))
col3.write(SR28)
col3.write('---')

XM = df_cut[['CC_Quartz',
 'CC_Hematite',
 'CC_Magnetite',
 'CC_Microcline',
 'CC_Anorthite',
 'CC_Anorthoclase',
 'CC_Muscovite',
 'CC_Amorphous',
 'Percent_CK',
'VMD'
]]

ym1= df_cut[[
'28DS',
]]

ym2= df_cut[[
'1DS',
]]
ym3= df_cut[[
'w/b',
]]

y_predictwb = rdfwb.predict(XM)
mean_absolute_errorwb = mean_absolute_error(ym3, y_predictwb)
st.markdown('This model (w/b) has mean absolute error = {:,.5f}'.format(mean_absolute_errorwb))

y_predict1D = rdf1D.predict(XM)
mean_absolute_error1D = mean_absolute_error(ym2, y_predict1D)
st.markdown('This model (1 Days Strength) has mean absolute error = {:,.2f} ksc'.format(mean_absolute_error1D))

y_predict = rdf.predict(XM)
mean_absolute_error = mean_absolute_error(ym1, y_predict)
st.markdown('This model (28 Days Strength) has mean absolute error = {:,.2f} ksc'.format(mean_absolute_error))

##############################################1DS surface


CC_Quartz = np.full((100,1),df['CC_Quartz'].iloc[0])                   
CC_Hematite = np.full((100,1),df['CC_Hematite'].iloc[0])      
CC_Magnetite = np.full((100,1),df['CC_Magnetite'].iloc[0] )     
CC_Microcline = np.full((100,1),df['CC_Microcline'].iloc[0])      
#CC_Anorthite = np.full((100,1),df['CC_Anorthite'].iloc[0]  )    
CC_Anorthoclase = np.full((100,1),df['CC_Anorthoclase'].iloc[0] )    
CC_Muscovite = np.full((100,1),df['CC_Muscovite'].iloc[0] )     
CC_Amorphous = np.full((100,1),df['CC_Amorphous'].iloc[0])      
Percent_CK = np.full((100,1),df['Percent_CK'].iloc[0] )     
#VMD =  np.full((100,1),df['VMD'].iloc[0]  )    

min_temp = 5
max_temp = 20

min_pressure = 0
max_pressure = 20


VMD, CC_Anorthite = np.meshgrid(np.linspace(min_temp, max_pressure, 10), np.linspace(min_pressure, max_temp, 10))
predict_x = np.concatenate((VMD.reshape(-1, 1), 
                           CC_Anorthite.reshape(-1, 1)), 
                           axis=1)
VMD0 = VMD.reshape(-1,1)

CC_Anorthite0 = CC_Anorthite.reshape(-1,1)

Surface_matrix = np.concatenate((CC_Quartz, CC_Hematite, CC_Magnetite, CC_Microcline,
       CC_Anorthite0, CC_Anorthoclase, CC_Muscovite, CC_Amorphous,
       Percent_CK, VMD0),  axis=1)
predict_1DS = rdf1D.predict(Surface_matrix)
VMD2 = VMD.reshape(1,-1)
CC_Anorthite2 = CC_Anorthite.reshape(1,-1)
x = VMD
y1 = CC_Anorthite
z=  predict_1DS.reshape(VMD.shape)

#z= predict_28DS.reshape(predict_pressure.shape)

fig8 = go.Figure(data=[go.Surface(x=x, y=y1, z=z, colorscale='Viridis',showscale = False)])




fig8.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=2, z=1.25)
)

fig8.update_layout(scene_camera=camera)    
fig8.update_layout(scene = dict(
                    xaxis_title='VMD',
                    yaxis_title='CC_Anorthite',
                    zaxis_title='predict_1DS'),
                    width=400,height=350,
                    margin=dict(r=5, b=0, l=0, t=0))

col7.write(fig8)


# Heatmap
if col4.button('Intercorrelation Heatmap'):
    #col4.header('Intercorrelation Matrix Heatmap')
    
    dff = pd.concat([X, y,y2,y3], axis=1)
    corr = dff.corr()
    f, ax = plt.subplots(figsize=(15, 15))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)
    col5.pyplot(f)
st.write('---')    
if col4.button('See Feature Importance (28Days Strength)'):
    #col5.header('Feature Importance - 28 Days strength prediction')    
    explainer = shap.TreeExplainer(rdf)
    shap_values = explainer.shap_values(X)

    f, ax = plt.subplots(figsize=(15, 15))
    plt.title('Feature importance based on SHAP values - 28 Days strength')
    shap.summary_plot(shap_values, X)
    col5.pyplot(f,bbox_inches='tight')
    
if col4.button('See Feature Importance (1 Days Strength)'):
    #col5.header('Feature Importance - 1 Days strength prediction')    
    explainer = shap.TreeExplainer(rdf1D)
    shap_values = explainer.shap_values(X2)

    f, ax = plt.subplots(figsize=(15, 15))
    #plt.title('Feature Importance - 1 Days strength prediction')
    shap.summary_plot(shap_values, X2)
    col5.pyplot(f,bbox_inches='tight')
    
if col4.button('See Feature Importance (w/b)'):
    #col5.header('Feature Importance - 1 Days strength prediction')    
    explainer = shap.TreeExplainer(rdfwb)
    shap_values = explainer.shap_values(X3)

    f, ax = plt.subplots(figsize=(15, 15))
    #plt.title('Feature Importance - 1 Days strength prediction')
    shap.summary_plot(shap_values, X3)
    col5.pyplot(f,bbox_inches='tight')








