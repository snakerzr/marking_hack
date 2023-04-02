import pickle

import pandas as pd

import matplotlib.pyplot as plt 

import plotly.express as px
import plotly.graph_objs as go

import streamlit as st

from ts_prediction import *
from anomaly import * 


tab1, tab2 = st.tabs(['Time Series prediction','Anomaly detection'])

with tab1:

    # Define the options for the first select box
    option1 = ['all', '1248F88441BCFC56', '289AEBCA82877CB1']

    # Define the options for the second select box
    option2 = ['all', '77', '50']

    # Define the options for the third select box
    option3 = [1, 2, 3, 4, 5, 6]

    # Create a container to hold the three select boxes in a row
    col1, col2, col3 = st.columns(3)

    # Add the first select box to the container
    with col1:
        gtin = st.selectbox('Select gtin', option1, disabled=False)

    # Add the second select box to the container
    with col2:
        region = st.selectbox('Select region', option2, disabled=False)

    # Add the third select box to the container
    with col3:
        months = st.selectbox('Select months to predict', option3, index=4, disabled=False)

    
    # st.write([gtin,region])
    

    end_dict = pd.DataFrame(generate_combs(option1,option2)).T

    # st.write(end_dict)
    model_path = end_dict.loc[(region,gtin),0]
    data_path = end_dict.loc[(region,gtin),1]
    params = end_dict.loc[(region,gtin),2]

    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)


    # 'Main procedure: all default, other options are dummy'
    df = pd.read_csv(data_path)  
    # df

    preds = predict_timeseries(model, df, months,*params)

    st.table(pd.DataFrame(preds,columns=['prediction'],index=[x for x in range(months)]))


    x_axis = pd.date_range(start='22/11/2021', freq='1M', periods=13)
    x_axis_2 = pd.date_range(start='30/11/2022', freq='1M', periods=months)
    # preds.insert(0, float(df['total'].tail(1)))
    df = df.append({'total':preds[0]}, ignore_index=True)

    fig = px.line(x=x_axis, y=df['total'].tail(13), markers=True)
    

    fig.add_trace(go.Scatter(
       x=x_axis,
       y=df['total'].tail(13),
       line=dict(color="#636EFA"),
       name="True"))

    fig.add_trace(go.Scatter(
       x=x_axis_2,
       y=preds,
       line=dict(color="orange", dash='dot'),
       name="Predicted",
       mode='lines+markers'))


    fig.update_layout(
       title="Outlet Distribution",
       xaxis_title="date",
       yaxis_title="sum",
       legend_title="legend",
       font=dict(family="Arial", size=20)
    )

    # fig.show()
    st.plotly_chart(fig, use_container_width=True)

    
with tab2:    
    st.header(header)
    st.caption(subheader)
    with st.expander('Для кого, польза для экономики, польза для "людей"'):
        st.markdown(application_fields)
        st.markdown(economy_benefit)
        st.markdown(consumer_benefit)
    
    with st.expander('Признаки коротко'):
        st.markdown(features_short)
    
    
    with st.expander("Гипотеза"):
        st.markdown(concept)
        
    st.header('Основная часть/функционал')
        
    prid_gtin = pd.read_csv('prid_gtin.csv',index_col=0)
    st.write('Для демонстрации есть 15 уникальных пар `prid`-`gtin`')
    col1, col2 = st.columns(2)
    prid_gtin_selected = col1.selectbox('Select prid+gtin',options=[x for x in range(15)])

    prid = prid_gtin.iloc[prid_gtin_selected,0]
    gtin = prid_gtin.iloc[prid_gtin_selected,1]

    st.metric('prid',prid)
    st.metric('gtin',gtin)
    
    csv1 = pd.read_csv('1.csv',index_col=0)
    csv2 = pd.read_csv('2.csv',index_col=0)
    csv3 = pd.read_csv('3.csv',index_col=0)

    df = preprocess(prid,gtin,csv1,csv2,csv3, fillna=True)
    mad_outliers = detect_outliers_on_residuals_with_mad(df)
    iforest_proba,iforest_outliers = isolation_forest_pred(df)
    _,q_df = quantile_transform(df)
    ensemble_outliers,ensemble_mean_scores, ensemble_median_scores = train_ensemble(estimators,q_df,threshold=0.75)
    
    with st.expander('Описание предобработки данных и код'):
        st.markdown(preprocessing_description)
        st.code(preprocessing_code)
    
    st.header('General info')
    st.write('Здесь может выводиться какая-то общая информация о паре `prid`-`gtin`')
    st.write('Кол-во записей в датасете: '+str(len(df)))
    
    
    st.header('MAD on seasonal decomposition residuals')
    
    with st.expander('Описание метода и код'):
        st.markdown(mad_description)
        st.code(mad_code)
    
    mad_threshold = st.slider('Number of min anomaly features',1,5,step=1,value=4)
    
    st.caption('В каких признаках аномалия')
    mad_outliers[mad_outliers['sum_resid_mad']>=mad_threshold]
    st.caption('Строки с аномалиями')
    df.iloc[mad_outliers[mad_outliers['sum_resid_mad']>3].index]
    
    st.header('Isolation forest')
    
    with st.expander('Описание метода и код'):
        st.markdown(iforest_description)
        st.code(iforest_code)
        
    if_threshold = st.slider('Isolation Forest probability threshold',0.5,1.,step=0.01,value=0.75)
    st.caption('Строки с аномалиями')
    df.loc[iforest_proba>if_threshold]
    
    st.header('Ensemble outliers')
    
    with st.expander('Описание метода и код'):
        st.markdown(ensemble_description)
        st.code(ensemble_code)    
    
    st.caption('Модели в ансамбле')
    for model in estimators:
        st.write(model)
    mode = st.selectbox('Select score averaging mode',['mean','median'])
    en_threshold = st.slider('Ensemble probability threshold',0.5,1.,step=0.01,value=0.75)
    st.caption('Строки с аномалиями')
    if mode == 'mean':
        df.loc[ensemble_mean_scores>en_threshold]
    elif mode == 'median':
        df.loc[ensemble_median_scores>en_threshold]
        
    with st.expander('Масштабируемость, улучшения и UI/UX'):
        st.markdown(scalability)
        st.markdown(improvements)
        st.markdown(ui_ux)
    
    