import streamlit as st
import joblib
from src.pipeline import ROOT,predict
st.set_page_config(page_title='YouTube Popularity Predictor',page_icon='📺',layout='centered')
st.title('📺 YouTube Video Popularity Prediction')
st.write('Estimate the number of likes for an existing video using views, duration, and number of tags.')
st.info('Educational regression model. Views are only known after publication; predictions are not guarantees.')
model_path=ROOT/'models'/'random_forest.joblib'
if not model_path.exists():
    st.error('Model not trained yet. Run: python train.py')
    st.stop()
model=joblib.load(model_path)
views=st.number_input('Video views',min_value=0,value=10000,step=1000)
duration=st.number_input('Video duration (seconds)',min_value=1,value=300)
tags=st.number_input('Number of tags',min_value=0,value=5)
if st.button('Predict popularity',type='primary'):
    result=predict(model,views,duration,tags)
    st.metric('Estimated likes',f'{result:,}')
    if views>0: st.write(f'Estimated like-to-view ratio: {result/views*100:.2f}%')
    st.caption('Interpret estimates cautiously, especially for inputs outside the training data range.')
metrics=ROOT/'outputs'/'metrics.csv'
if metrics.exists():
    import pandas as pd
    st.subheader('Model evaluation')
    st.dataframe(pd.read_csv(metrics),hide_index=True)
