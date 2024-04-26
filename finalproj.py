import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


#downloading files
X_train=pd.read_csv(r"C:\Users\kbrad\Downloads\p1_train (1).csv")
X_test=pd.read_csv(r"C:\Users\kbrad\Downloads\p1_test (1).csv")

#Assigning column names

X_test =X_test.rename(columns={'1.589300268390259419e+01': 't_coln1', '1.171282902260990966e+01': 't_coln2','-3.756792885773750612e+01':'test_target'})

X_train =X_train.rename(columns={'-7.262173392018990370e+00': 't_coln1', '9.572603824406265005e+00': 't_coln2','5.358725498169498280e+00':'train_target'})

#fixing sample data
X_train=X_train.head(4000)
X_test=X_test.head(4000)

#Seperating feature and target values
x_train=X_train.drop(['train_target'],axis=1)
y_train=X_train['train_target']	
x_test=X_test.drop(['test_target'],axis=1)
y_test=X_test['test_target']

#Scalling

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)

#Model selection and predict target values

model=LinearRegression()
model.fit(x_train_scaled,y_train)
train_pred=model.predict(x_train_scaled)
test_pred=model.predict(x_test_scaled)
r2=r2_score(y_test,test_pred)

#Streamlit part
import streamlit as st

st.set_page_config(layout='wide')
st.title(":green[ PREDICT THE CURRENT HEALTH OF AN ORGANISM]")

with st.form("Data"):
        col1,col2=st.columns(2)
        with col1:
                st.write("PREDICT THE HEALTH MEASURES USING FEATURE VALUES")
                t_coln1= st.text_input("Enter first feature value(Min:-19.981237631733517,Max:19.99707715064372)")
                t_coln2= st.text_input("Enter second feature value(Min:-19.9771347553298,Max:19.962176595929364)")
                submit_button=st.form_submit_button(label="PREDICT HEALTH MEASURES")

                if submit_button:
                        import pickle
                        with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\final_model.pkl",'rb') as file:
                            loaded_model=pickle.load(file)

                        with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\final_scaler.pkl",'rb') as file1:
                            loaded_scalar=pickle.load(file1)
                        
                        user_input=np.array([[t_coln1,t_coln2]])

                        user_input1=scaler.transform(user_input)
                        user_prediction=model.predict(user_input1)
                        st.write("Predicted Health measures:",user_prediction)
                        if user_prediction>0:
                              st.write("RESULT IS IN THE AVERAGE VALUE")
                        else:
                              st.write("RESULT IS LESSER THAN THE AVERAGE VALUE")

with st.sidebar:
    st.title(":blue[PREDICTION OF HEALTH MEASURES]")
    st.header(":red[STEPS FOLLOWED]")
    st.caption("Downloaded the data set and detected outliers in the dataset")
    st.caption("Transformed the data into a suitable format and perform any necessary cleaning and pre-processing steps")
    st.caption("Created ML Regression model which predicts continuous variable in the target column’")
    
    st.caption("Created a streamlit page where you can insert each column value and you will get predicted health measures")
   
    st.header(":red[TECHNOLOGIES USED]")
    st.caption("Python scripting,Pandas,Numpy,Seaborn,Matplotlib,Data Preprocessing,EDA, Streamlit")





