import pandas as pd

def generate_combs(option1,option2):
    model_name = []
    data_name = []
    combs = []

    for x in option1:
        for y in option2:
            combs.append((y,x))
            mn = 'models/' + 'model_' + str(y) + '_' + str(x) + '.pkl'
            dn = 'data/' + 'data_' + str(y) + '_' + str(x) + '.csv'
            model_name.append(mn)
            data_name.append(dn)


    params = [
                [25,5,5,5],
                [15,0,5,5],
                [15,0,5,5],
                [15,0,5,5],
                [15,5,3,3],
                [15,0,5,5],
                [15,0,5,5],
                [25,5,5,5],
                [15,0,5,5]]

    end_dict = {}    
    for i in range(len(combs)):
        end_dict[combs[i]] = [model_name[i],data_name[i],params[i]]
        
    return end_dict

def add_features(data, amount_of_lags=1, amount_of_sma=6, rolling_window=4, rolling_step=3):
    modified_data = data.copy()

    for i in range(1, amount_of_lags+1):
        modified_data['lag_{}'.format(i)] = modified_data['total'].shift(i)


    for i in range(amount_of_sma):
        col_name = 'moving_average_{}'.format(rolling_window+rolling_step*i)
        modified_data[col_name] = modified_data['total'].shift().rolling(rolling_window+rolling_step*i).mean()

    modified_data = modified_data.dropna()

    return modified_data
    



def predict_timeseries(model, data, amount_of_preds=3, amount_of_lags=25, amount_of_sma=5, rolling_window=5, rolling_step=5):

    temp_data = data.reset_index(drop=True)
    preds = []

    for i in range(amount_of_preds):
        temp_data = temp_data.append({'total':0}, ignore_index=True)
        lagged_data = add_features(temp_data, amount_of_lags, amount_of_sma, rolling_window, rolling_step)
        prediction = model.predict(lagged_data.drop('total', axis=1).tail(1))
        temp_data['total'].iloc[-1] = prediction
        preds.append(float(prediction))

    return preds



header_ = 'Сервис для прогнозирования спроса'

general_info_ = '''Cервис позволяет прогнозировать спрос определенных категорий товаров в определенных регионах. Выполнен в минималистичном дизайне с максимально понятным UI. Можно расчитать спрос на определенный продукт в определенном регоине на выбранное количество месяцев. 

Доступно три опции для gtin и 3 для регионов. 
`ALL` в gtin это 5 самый богато представленных gtin в выборке данных. 
`ALL` в регионах это 3 самых богато представленных региона в выборке. 
А также можно выбрать какой-то специфический gtin и регион.'''

goverment_benefit_ = '''### Польза для государства:

* инструментом для планирования бюджета и оптимизации закупок.
* избежать переплаты за товары, которые не будут использованы
* избежать дефицита товаров, которые необходимы для реализации государственных программ и проектов'''

business_benefit_ = '''### Польза для бизнеса:

* Оптимизировать производственные процессы и управлять запасами товаров, чтобы избежать недостатка или избытка товаров на складах. 
* Разрабатывать маркетинговые стратегии, которые соответствуют потребностям потребителей и позволяют продавать товары более эффективно. 
* Принимать решения о введении новых товаров на рынок или о снятии с производства старых товаров. 
* Предотвращать потери бизнеса, связанные с невостребованными товарами или недостатком товаров на складах.'''

scalability_ = '''### Масштабируемость:

Gtin очень много и в нашем сервмсе мы использовали всего несколько самых часто встречающихся, поэтому при желании можно расширить сервис на все gtinы и регионы.'''

improvements_ = '''### Потенциальные улучшения:
Мы использовали линейную регрессию и функцию для генерации лагов,
можно попробовать использовать другие модели такие как: arima, sarima, prophet, lstm и другие. 

Еще если расширить, можно изучить сезонность и это тоже поможет качественней прогнозировать спрос.

Можно снизить потенциальный риск ошибки модели, добавив доверительный интервал и уверенность модели.'''
