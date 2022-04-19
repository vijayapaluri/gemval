import streamlit as st
import gemval
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait


    






#st.title("Gemval Index")

st.markdown("<h1 style='text-align: center; color:#F08080 ;'>Gemval Index</h1>", unsafe_allow_html=True)
Models=st.sidebar.selectbox("Models", ["ARIMA","LSTM","EXPO"])
period=st.sidebar.selectbox("period", ["6 months","1 year","2 years"])  
if st.sidebar.button("Refresh"):
    try:
        url="https://gemval.com/gva/?index=GVA&term=2"
        #wait = WebDriverWait(driver, 30)
        driver=webdriver.Firefox(executable_path=GeckoDriverManager().install())
        driver = webdriver.Firefox(executable_path='C:/Users/Vijaya/.wdm/drivers/geckodriver/win64/v0.31.0/geckodriver-v0.31.0-win64/geckodriver.exe')
     
        wait = WebDriverWait(driver, 30)
        driver.get(url)
        
        driver.execute_script("document.getElementById('gemval-aggregate-chart').scrollIntoView()")

    # wait until the chart div has been rendered before accessing the data provider
        wait.until(lambda x: x.find_element_by_class_name("amcharts-chart-div").is_displayed())
        time.sleep(5)
        temp=driver.execute_script("return AmCharts.charts[0].dataProvider")
        df=pd.DataFrame(temp)
        df.set_index("date",inplace=True)
        driver.close()
    except Exception as e:
        print(e)
    st.markdown("<h3 style='text-align: Left; color:  #CA6F1E;'>Dataset</h3>", unsafe_allow_html=True)
    st.write(df)

    gem_val,train,test,train_log,test_log= gemval.dataset(df)
    st.markdown("<h3 style='text-align: Left; color:  #CA6F1E;'>Summary Statistics</h3>", unsafe_allow_html=True)
    st.dataframe(gem_val.describe())
    st.dataframe(gem_val.skew())
    st.dataframe(gem_val.kurt())

    st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Before Predictions on Train values</h5>", unsafe_allow_html=True)

    fig=plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_facecolor('#EAF2F8')
    plt.plot(train_log)
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend(['actual','values'])
    st.pyplot(fig)
    
    if Models=="ARIMA":
        
        if period=="6 months":
      

        

            y_gemval_6m,y_pred_df_gemval_6m2,y_fit=gemval.arima_6m(train_log,test_log)
            st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 6 months</h5>", unsafe_allow_html=True)

            fig=plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_facecolor('#EAF2F8')
            plt.title("Confidence Interval after 6 months")
            plt.plot(y_gemval_6m)
            plt.plot(y_pred_df_gemval_6m2["Predictions"],color = 'red',label='predicted')
            plt.plot(y_pred_df_gemval_6m2['lower value'], linestyle = '--', color = 'red', linewidth = 0.5, label='lower ci')
            plt.plot(y_pred_df_gemval_6m2['upper value'], linestyle = '--', color = 'red', linewidth = 0.5, label='upper ci')

            plt.fill_between(y_pred_df_gemval_6m2["Predictions"].index.values,y_pred_df_gemval_6m2['lower value'], y_pred_df_gemval_6m2['upper value'], color = 'grey', alpha = 0.2)
            plt.legend(loc = 'best')
            st.pyplot(fig)


            st.write(y_fit)
        elif period=="1 year":



            st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 1 year</h5>", unsafe_allow_html=True)

            y_test,y_pred,y_fit=gemval.arima_1year(gem_val)

            fig=plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_facecolor('#EAF2F8')
            plt.plot(y_test)
            plt.title("Confidence Interval after 1 year")
            plt.plot(y_pred["Predictions"],color = 'red',label='predicted')
            plt.plot(y_pred['lower value'], linestyle = '--', color = 'red', linewidth = 0.5, label='lower ci')
            plt.plot(y_pred['upper value'], linestyle = '--', color = 'red', linewidth = 0.5, label='upper ci')
            plt.fill_between(y_pred["Predictions"].index.values,
                             y_pred['lower value'], 
                             y_pred['upper value'], 
                             color = 'grey', alpha = 0.2)
            plt.legend(loc = 'best')
            st.pyplot(fig)

            st.write(y_fit)
        elif period=="2 years":
        
            st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 2 year</h5>", unsafe_allow_html=True)
            y_test,y_pred,y_fit=gemval.arima_2years(gem_val)
            fig = plt.figure(figsize = (16,8))
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_facecolor('#EAF2F8')
            plt.title("Confidence Interval after 2 years")
            plt.plot(y_test)
            plt.plot(y_pred["Predictions"],color = 'red',label='predicted')
            plt.plot(y_pred['lower value'], linestyle = '--', color = 'red', linewidth = 0.5, label='lower ci')
            plt.plot(y_pred['upper value'], linestyle = '--', color = 'red', linewidth = 0.5, label='upper ci')
            plt.fill_between(y_pred["Predictions"].index.values,
                             y_pred['lower value'], 
                             y_pred['upper value'], 
                             color = 'grey', alpha = 0.2)
            plt.legend(loc = 'best')
            st.pyplot(fig)

            st.write(y_fit)
    elif Models=="LSTM": 
       
        if period=="6 months":
                y_test,y_pred,eval=gemval.LSTM_6months(gem_val)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>LSTM Forecast Predictions next 6 months</h5>", unsafe_allow_html=True)
                plt.plot(y_test,label='actual')
                plt.plot(y_pred,label='predicted')
                plt.legend(loc = 'best')
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(eval)
              
        elif period=="1 year":
                y_test,y_pred,eval=gemval.LSTM_1year(gem_val)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>LSTM Forecast Predictions next 1 year</h5>", unsafe_allow_html=True)
                plt.plot(y_test,label='actual')
                plt.plot(y_pred,label='predicted')
                plt.legend(loc = 'best')
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(eval)
                
                
        elif period=="2 years":
                y_test,y_pred,eval=gemval.LSTM_2years(gem_val)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>LSTM Forecast Predictions next 2 years</h5>", unsafe_allow_html=True)
                plt.plot(y_test,label='actual')
                plt.plot(y_pred,label='predicted')
                plt.legend(loc = 'best')
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(eval)
                
                
    elif Models=="EXPO": 
       
        if period=="6 months":
                gemval_test_6m,gemval_pred1_6m,gemval_train_6m,eVal=gemval.EXPO_6months(gem_val)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>EXPO Forecast Predictions next 6 months</h5>", unsafe_allow_html=True)
               
                plt.plot(gemval_train_6m, label='Train')
                plt.plot(gemval_test_6m, gemval_pred1_6m, label='Exponential Smoothing ')
                plt.legend(loc='best')
                plt.show()
                
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(eVal)
        elif period=="1 year":
                gemval_Test_1y,gemval_pred1_1y, gemval_Train_1y,eVal=gemval.EXPO_1y(gem_val)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>EXPO Forecast Predictions next 1 year</h5>", unsafe_allow_html=True)
                plt.rcParams["figure.figsize"] = [16,9]
                plt.plot(gemval_Train_1y, label='Train')
                plt.plot(gemval_Test_1y, gemval_pred1_1y, label='Exponential Smoothing ')
                plt.legend(loc = 'best')
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(eVal)
              
        elif period=="2 years":
                gemval_Test_2y,gemval_pred1_2y,gemval_Train_2y,eVal=gemval.EXPO_2y(gem_val)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>EXPO Forecast Predictions next 1 year</h5>", unsafe_allow_html=True)
                plt.rcParams["figure.figsize"] = [16,9]
                plt.plot(gemval_Train_2y, label='Train')
                plt.plot(gemval_Test_2y, gemval_pred1_2y, label='Exponential Smoothing ')
                plt.legend(loc = 'best')
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(eVal)             