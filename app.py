import streamlit as st
import pickle
import numpy as np

df= pickle.load(open('dataframe.pkl','rb'))
pipe=pickle.load(open("pipe_model.pkl",'rb')) 
st.title("Laptop Price preditor")
st.header("ML Model")
company=st.selectbox('Please Choose Your Brand',df['Company'].unique(),index=4)
Type = st.selectbox('Please Choose Your Type',df['TypeName'].unique())
Cpu	= st.selectbox('Please Choose Your Cpu',df['Cpu'].unique())
Ram= st.selectbox('Please Choose Your Ram',df['Ram'].unique())
Gpu= st.selectbox('Please Choose Your Gpu',df['Gpu'].unique())
OpSys= st.selectbox('Please Choose Your OpSys',df['OpSys'].unique(),index=2)
Weight= st.slider('Please Choose Your Weight',min_value=0.5,max_value=4.8,value=2.0,step=0.2)
Touchscreen= st.selectbox('Please Choose Your Touchscreen',['yes','no'])
IPS= st.selectbox('Please Choose Your IPS',['yes','no'])
ppi= st.slider('Please Choose Your ppi',min_value=90,max_value=350,value=220,step=10)
if st.button("PREDICT PRICE"):
  query=np.array([[Company, TypeName, Cpu, Ram, Gpu, OpSys, Weight,
       Touchscreen, IPS, ppi]])
  op=pipe.predict(query)
  st.subheader(round(np.exp(op[0])))
