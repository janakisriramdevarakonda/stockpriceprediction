import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pywt

st.title("Stock Price Prediction ")
uploaded_file = st.sidebar.file_uploader("Upload Stock dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())
    
    date_column = st.sidebar.selectbox("Select date column", df.columns)
    target_column = st.sidebar.selectbox("Select target (stock price) column", df.columns)

    #moving avg
    df['MA100'] = df['Close'].rolling(100).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA100'], mode='lines', name='100-day MA'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], mode='lines', name='200-day MA'))
        
    fig.update_layout(title='Stock Closing Price with Moving Averages',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=True)
        
    st.plotly_chart(fig)

    
    #choose prediction interval
    prediction_interval = st.sidebar.slider("Select Prediction Interval (days)", min_value=10, max_value=365, value=30)
    
    if date_column and target_column:

        # Check if required columns exist
        required_columns = ['Open', 'High', 'Low', 'Volume']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Dataset must contain {required_columns} columns.")
        else:
            # Prepare data
            close_prices = df[target_column].values
            wavelet = 'db4'  # Daubechies wavelet
            coeffs = pywt.wavedec(close_prices, wavelet, level=3)

            # Reconstruct time series using selected wavelet coefficients
            reconstructed = pywt.waverec(coeffs[:2], wavelet)
            reconstructed_padded = np.pad(reconstructed, (0, len(df) - len(reconstructed)), 'edge')

            # Add Wavelet feature
            df['Wavelet_Close'] = reconstructed_padded

            # Define features and target
            fea = ['Wavelet_Close', 'Open', 'High', 'Low', 'Volume']
            X1 = df[['Wavelet_Close', 'Open', 'High', 'Low', 'Volume']]
            y1 = df[target_column]

        # Split data
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
        
        #Wavelet training
        xgb_model1 = xgb.XGBRegressor(base_score=0.5,
                                     booster='gbtree',
                                     n_estimators=330,
                                     objective='reg:squarederror',
                                     max_depth=3,
                                     learning_rate=0.01)

        xgb_model1.fit(X_train1, y_train1,
                      eval_set=[(X_train1, y_train1), (X_test1, y_test1)],
                      verbose=100)

        # Predict and Evaluate(Wavelet)
        y_pred1 = xgb_model1.predict(X_test1)
        mae1 = mean_absolute_error(y_test1, y_pred1)
        rmse1 = np.sqrt(mean_squared_error(y_test1, y_pred1))
        r21 = r2_score(y_test1, y_pred1)        

        #Make predictions (WaveXGB)
        last_row1 = df.iloc[-1][fea].copy()
        future_preds1 = []
        future_dates1 = pd.date_range(start=df[date_column].max(), periods=prediction_interval+1).tolist()[1:]
        
        # Future Prediction
        
        future_X = X1.tail(prediction_interval).fillna(method='bfill')
        future_predictions1 = xgb_model1.predict(future_X)

        # Format Future Predictions Table
        # Ensure the index is in datetime format
        df.index = pd.to_datetime(df.index)
        start_date = df.index[-1] + pd.Timedelta(days=1)
        future_dates1 = pd.date_range(start=df[date_column].max(), periods=prediction_interval+1).tolist()[1:]
        
        future_df1 = pd.DataFrame({'ds': future_dates1, 'Predicted': future_predictions1})
        future_df1['ds'] = future_df1['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime
        future_df1['Predicted'] = future_df1['Predicted'].apply(lambda x: round(x, 4))  # Round to 4 decimal places
        
        #Preparing data(FBP)
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(by=[date_column])
        prophet_df = df[[date_column, target_column]].rename(columns={date_column: "ds", target_column: "y"})
        
        #Preparing data(XGB)
        df['day'] = df[date_column].dt.day
        df['month'] = df[date_column].dt.month
        df['year'] = df[date_column].dt.year
        df['price_lag_1'] = df[target_column].shift(1).fillna(method='bfill')
        
        features = ['day', 'month', 'year', 'price_lag_1']
        X = df[features]
        y = df[target_column]

        #training
        train_size = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        dates_test = df[date_column].iloc[train_size:]
        
        #train XGB
        xgb_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=800,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
        
        xgb_model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)
        
        y_pred_xgb = xgb_model.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
        xgb_r2 = r2_score(y_test, y_pred_xgb) * 100
        xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

       
        #Make predictions (XGB)
        last_row = df.iloc[-1][features].copy()
        future_preds = []
        future_dates = pd.date_range(start=df[date_column].max(), periods=prediction_interval+1).tolist()[1:]
        
        for future_date in future_dates:
            last_row['day'] = future_date.day
            last_row['month'] = future_date.month
            last_row['year'] = future_date.year
            prediction = xgb_model.predict(pd.DataFrame([last_row]))[0]
            future_preds.append(prediction)
            last_row['price_lag_1'] = prediction  # Update for next iteration
        
        future_df = pd.DataFrame({"ds": future_dates, "Predicted": future_preds})
        
        #train FBP
        prophet_model = Prophet()
        prophet_model.fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=prediction_interval)  # Predict user-selected days
        forecast = prophet_model.predict(future)
        
        #accuracy for Prophet (using last known values as reference)
        test_prophet = prophet_df[-prediction_interval:]
        forecast_prophet = forecast[-prediction_interval:]
        prophet_mae = mean_absolute_error(test_prophet['y'], forecast_prophet['yhat'])
        prophet_r2 = r2_score(test_prophet['y'], forecast_prophet['yhat']) * 100
        prophet_rmse = np.sqrt(mean_squared_error(test_prophet['y'], forecast_prophet['yhat']))
        
        xgboost, wavexg, fbprophet = st.tabs(["XGBoost", "Wavelet transform + XGBoost",  "FBProphet"])
        
        with xgboost:
            st.subheader("XGBoost Predictions")
            fig_xgb, ax_xgb = plt.subplots(figsize=(10, 4))
            ax_xgb.plot(dates_test, y_test.values, label="Actual Prices", color='blue')
            ax_xgb.plot(dates_test, y_pred_xgb, label="Predicted Prices", color='red')
            ax_xgb.legend()
            ax_xgb.set_xlabel("Date")
            ax_xgb.set_ylabel("Stock Price ")
            st.pyplot(fig_xgb)
            
            #prediction Plot
            fig_future_xgb, ax_future_xgb = plt.subplots(figsize=(10, 4))
            ax_future_xgb.plot(future_df['ds'], future_df['Predicted'], label="Future Predictions ", color='red')
            ax_future_xgb.legend()
            ax_future_xgb.set_xlabel("Date")
            ax_future_xgb.set_ylabel("Stock Price")
            ax_future_xgb.set_title("Future Predictions using XGBoost")
            st.pyplot(fig_future_xgb)
            
            st.write("Mean Absolute Error:", xgb_mae)
            st.write("R-squared Score (Accuracy %):", xgb_r2, "%")
            st.write("Root Mean Squared Error:", xgb_rmse)
            
            st.write("Actual vs Predicted:")
            xgb_results = pd.DataFrame({"Date": dates_test.values, "Actual": y_test.values, "Predicted": y_pred_xgb})
            st.write(xgb_results.tail(10))
            
            st.write("Future Predictions:")
            st.write(future_df[['ds', 'Predicted']].tail(10))

        with wavexg:
            
            # Plot Predictions
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df.index[-len(y_test1):], y_test1, label="Actual Price ", color='blue')
            ax1.plot(df.index[-len(y_test1):], y_pred1, label="Predicted Price ", color='red')
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Stock Price")
            ax1.set_title("Stock Price Prediction using Wavelet-XGBoost")
            ax1.legend()
            st.pyplot(fig1)    

            st.subheader("Model Performance")
            st.write(f"Mean Absolute Error: {mae1:.4f}")
            st.write(f"R-Squared Score(Accuracy %): {r21:.2%}")
            st.write(f"Root Mean Squared Error: {rmse1:.4f}")
            
            
            st.subheader("Future Predictions")
            st.write(future_df1.tail(130)) 
            #st.write(future_df1)
            



        with fbprophet:
            st.subheader("Prophet Forecast")
            fig, ax = plt.subplots(figsize=(12, 6))
            prophet_model.plot(forecast, ax=ax)
            ax.legend(["Actual Data", "Predicted Data", "Uncertainity Intervel", "Upper Bound"])
            st.pyplot(fig)

            fig_zoom, ax_zoom = plt.subplots(figsize=(18, 4))
            forecast_zoom = forecast[-prediction_interval:]
            ax_zoom.plot(forecast_zoom['ds'], forecast_zoom['yhat'], label="Predicted", color='red')
            ax_zoom.fill_between(forecast_zoom['ds'], forecast_zoom['yhat_lower'], forecast_zoom['yhat_upper'], color='maroon', alpha=0.3, label="uncertainity Interval")
            ax_zoom.legend()
            ax_zoom.set_title("Zoomed Prophet Prediction for Selected Interval")
            st.pyplot(fig_zoom)
          
            st.write("Mean Absolute Error:", prophet_mae)
            st.write("R-squared Score (Accuracy %):", prophet_r2, "%")
            st.write("Root Mean Squared Error:", prophet_rmse)
            
            st.write("Predicted Values:")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))



       
