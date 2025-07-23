import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings
import joblib
from datetime import timedelta

warnings.filterwarnings("ignore")

#load data set
def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    return data


#model = joblib.load('models/xgboost_model.pkl')
features = ['Day before holiday','Non Holiday date',
       'Day after holiday', 'Non Holiday',
       'Payment Week', 'Weekend',
       'No. of competitors', 'Quarterly Average sale',
        'Montly Average sale', 'Qtrly sales std dev',
       'Monthly sale std dev', 'dayofweek', 'month', 'sin_dow', 'lag_1',
       'lag_7', 'lag_3', 'lag_5', 'rolling_mean_7']
target = ['Sales']

#forecast next week function
def forecast_next_week(data, model):
    from datetime import timedelta

    last_date = pd.to_datetime(data.reset_index()['Date'].max())
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['dayofweek'] = future_df['Date'].dt.dayofweek
    future_df['month'] = future_df['Date'].dt.month
    future_df['sin_dow'] = future_df['dayofweek'].apply(lambda x: np.sin(2 * np.pi * x / 7))
    future_df['Weekend'] = future_df['dayofweek'].isin([5,6]).astype(int)
    future_df['Day before holiday'] = 0
    future_df['Non Holiday date'] = 1
    future_df['Day after holiday'] = 0
    future_df['Non Holiday'] = 1
    future_df['Payment Week'] = future_df['Date'].dt.day.apply(lambda d: 1 if d <= 7 or (14 <= d <= 21) else 0)
    latest_values = data.iloc[-1]
    static_cols = [
     'No. of competitors',
    'Quarterly Average sale', 'Montly Average sale',
    'Qtrly sales std dev', 'Monthly sale std dev'
    ]
    for col in static_cols:
        future_df[col] = latest_values[col]
    history = data.reset_index()[['Date', 'Sales']].copy()
    history['Date'] = pd.to_datetime(history['Date'])
    preds = []
    for i in range(7):
        current_date = future_df.loc[i, 'Date']
        lag_1 = history.iloc[-1]['Sales']
        lag_3 = history.iloc[-3]['Sales']
        lag_5 = history.iloc[-5]['Sales']
        lag_7 = history.iloc[-7]['Sales']
        rolling_mean_7 = history['Sales'][-7:].mean()
        future_df.at[i, 'lag_1'] = lag_1
        future_df.at[i, 'lag_3'] = lag_3
        future_df.at[i, 'lag_5'] = lag_5
        future_df.at[i, 'lag_7'] = lag_7
        future_df.at[i, 'rolling_mean_7'] = rolling_mean_7
        input_row = future_df.loc[i, features].values.reshape(1, -1)
        prediction = model.predict(input_row)[0]
        preds.append(prediction)
        new_row = pd.DataFrame({'Date': [current_date], 'Sales': [prediction]})
        history = pd.concat([history, new_row], ignore_index=True)

    future_df['predicted_sale'] = preds
    future_df['predicted_sale'] = future_df['predicted_sale'].round(0).astype(int)
    future_df['Date'] = future_df['Date'].dt.date
    future_df['Day'] = pd.to_datetime(future_df['Date']).dt.day_name() 
    return future_df[['Date', 'Day', 'predicted_sale']]

def forecast_next_month(data, model):
    last_date = pd.to_datetime(data.reset_index()['Date'].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['dayofweek'] = future_df['Date'].dt.dayofweek
    future_df['month'] = future_df['Date'].dt.month
    future_df['sin_dow'] = future_df['dayofweek'].apply(lambda x: np.sin(2 * np.pi * x / 7))
    future_df['Weekend'] = future_df['dayofweek'].isin([5,6]).astype(int)
    future_df['Day before holiday'] = 0
    future_df['Non Holiday date'] = 1
    future_df['Day after holiday'] = 0
    future_df['Non Holiday'] = 1
    future_df['Payment Week'] = future_df['Date'].dt.day.apply(lambda d: 1 if d <= 7 or (14 <= d <= 21) else 0)

    latest_values = data.iloc[-1]
    static_cols = [
        'No. of competitors',
        'Quarterly Average sale', 'Montly Average sale',
        'Qtrly sales std dev', 'Monthly sale std dev'
    ]
    for col in static_cols:
        future_df[col] = latest_values[col]

    history = data.reset_index()[['Date', 'Sales']].copy()
    history['Date'] = pd.to_datetime(history['Date'])

    preds = []
    for i in range(30):
        current_date = future_df.loc[i, 'Date']
        # Compute lags from history, which includes previous predictions
        lag_1 = history.iloc[-1]['Sales']
        lag_3 = history.iloc[-3]['Sales'] 
        lag_5 = history.iloc[-5]['Sales'] 
        lag_7 = history.iloc[-7]['Sales'] 
        rolling_mean_7 = history['Sales'][-7:].mean()

        future_df.at[i, 'lag_1'] = lag_1
        future_df.at[i, 'lag_3'] = lag_3
        future_df.at[i, 'lag_5'] = lag_5
        future_df.at[i, 'lag_7'] = lag_7
        future_df.at[i, 'rolling_mean_7'] = rolling_mean_7

        input_row = future_df.loc[i, features].values.reshape(1, -1)
        prediction = model.predict(input_row)[0]
        preds.append(prediction)

        # Append prediction to history for future lags
        new_row = pd.DataFrame({'Date': [current_date], 'Sales': [prediction]})
        history = pd.concat([history, new_row], ignore_index=True)

    future_df['predicted_sale'] = preds
    final_forecast = future_df[['Date', 'predicted_sale']].copy()
    final_forecast['predicted_sale'] = final_forecast['predicted_sale'].round(0).astype(int)
    final_forecast['Day'] = pd.to_datetime(final_forecast['Date']).dt.day_name()
    return final_forecast[['Date', 'Day', 'predicted_sale']]


if __name__ == "__main__":
    file_path = "sales_with_extra_features4.csv"
    data = load_and_clean_data(file_path)

    model = joblib.load('models/xgboost_model.pkl')

    forecast = forecast_next_month(data, model)
    ##print("forecast for next 7 days")

    ##print(forecast)

