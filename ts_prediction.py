import pandas as pd

def add_features(data, amount_of_lags=1, amount_of_sma=6, rolling_window=4, rolling_step=3):
    modified_data = data.copy()

    for i in range(1, amount_of_lags+1):
        modified_data['lag_{}'.format(i)] = modified_data['total'].shift(i)


    for i in range(amount_of_sma):
        col_name = 'moving_average_{}'.format(rolling_window+rolling_step*i)
        modified_data[col_name] = modified_data['total'].shift().rolling(rolling_window+rolling_step*i).mean()

    modified_data = modified_data.dropna()

    return modified_data
    



def predict_timeseries(model, data, amount_of_preds=3):

    temp_data = data.reset_index(drop=True)
    preds = []

    for i in range(amount_of_preds):
        temp_data = temp_data.append({'total':0}, ignore_index=True)
        lagged_data = add_features(temp_data, 25, 5, 5, 5)
        prediction = model.predict(lagged_data.drop('total', axis=1).tail(1))
        temp_data['total'].iloc[-1] = prediction
        preds.append(float(prediction))
    #print(temp_data)
    return preds