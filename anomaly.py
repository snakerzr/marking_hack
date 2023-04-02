import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import QuantileTransformer


from pyod.models.mad import MAD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

# ф-ция "вытаскивания" данных о конкретном товаре и производителе из всех данных
def slice_and_transform(inn_prid,gtin,df,df1,df2, fillna=True):
    
    # in table
    temp_df_in = df.loc[(df['prid'] == inn_prid)&(df['gtin'] == gtin),['dt','cnt']]
    temp_df_in.columns = ['dt','cnt_in']
    temp_df_in['dt'] = pd.to_datetime(temp_df_in['dt'])
    temp_df_in = temp_df_in.set_index('dt')
    temp_df_in = temp_df_in.sort_index()
    temp_df_in = temp_df_in.resample('1D').sum()
    
    # move table
    temp_df_mv = df1.loc[(df1['prid'] == inn_prid)&(df1['gtin'] == gtin)] 
    temp_df_mv['dt'] = pd.to_datetime(temp_df_mv['dt'])
    temp_df_mv = temp_df_mv.set_index('dt')
    temp_df_mv = temp_df_mv.sort_index()
    resample_dict = {'gtin':'count','sender_inn':'nunique','receiver_inn':'nunique','cnt_moved':['mean','median','sum']}
    temp_df_mv = temp_df_mv.resample('1D').agg(resample_dict)
    temp_df_mv.columns = ['count','sender_inn_nunique','receiver_inn_nunique','cnt_moved_mean','cnt_moved_median','cnt_moved_sum']
    
    # out table
    temp_df_out = df2.loc[(df2['prid'] == inn_prid)&(df2['gtin'] == gtin),['dt','price','cnt']] 
    temp_df_out.columns = ['dt','price_out','cnt_out']
    temp_df_out['dt'] = pd.to_datetime(temp_df_out['dt'])
    temp_df_out = temp_df_out.set_index('dt')
    temp_df_out = temp_df_out.sort_index()
    temp_df_out = temp_df_out.resample('1D').agg({'price_out':['mean','median'],'cnt_out':'sum'})
    temp_df_out.columns = ['price_out_mean','price_out_median','cnt_out_sum']
    
    if fillna:
        temp_df_in = temp_df_in.fillna(0)
        temp_df_mv = temp_df_mv.fillna(0)
        temp_df_out = temp_df_out.fillna(0)
    
    temp_df = temp_df_in.join(temp_df_out,how='outer').join(temp_df_mv,how='outer')
    
    
    return temp_df


# Ф-ция добавления признаков
def enrich(df):
#     Create in out cumsum delta
    df['cnt_in_cumsum'] = df['cnt_in'].cumsum()
    df['cnt_out_cumsum'] = df['cnt_out_sum'].cumsum()
    df['in_out_delta'] = df['cnt_in_cumsum'] - df['cnt_out_cumsum']
    df['in_out_delta_stationary'] = df['in_out_delta'].diff()
    
#     Create sender_receiver_ratio
    df['sender_receiver_ratio'] = df['sender_inn_nunique'] / df['receiver_inn_nunique']
    df['sender_receiver_sum'] = df['sender_inn_nunique'] + df['receiver_inn_nunique']
    
    
    # df = df.fillna(0)
    
    return df


def preprocess(inn_prid,gtin,df,df1,df2, fillna=True):
    result = slice_and_transform(inn_prid,gtin,df,df1,df2, fillna=True)
    result = enrich(result)
    return result


# Ф-ция детекции аномалий по остаткам сезонной декомпозии временного ряда с помощью MAD
def detect_outliers_on_residuals_with_mad(df,cols=None):
    
    def mad_on_residuals(df,col_name):
        resid = seasonal_decompose(df[col_name].fillna(0)).resid.values.reshape(-1, 1)
        mad = MAD().fit(resid)
        is_outlier = mad.labels_ == 1
        # display(mad.threshold_)
        # display(mad.decision_scores_)
        return is_outlier
    
    if cols==None:
        cols = ['in_out_delta_stationary','price_out_mean', 'price_out_median',
                'cnt_moved_mean','cnt_moved_median', 'cnt_moved_sum','cnt_out_sum']
        

    outliers = pd.DataFrame(columns=cols)


    for col in cols:
        is_outlier = mad_on_residuals(df,col)
        outliers[col] = is_outlier
    
    cols = [x+'_resid_mad' for x in cols]
    outliers.columns = cols
    
    outliers['sum_resid_mad'] = outliers.sum(axis=1)
    
    return outliers



def add_date_features(df):
    df = df.copy()
    df['day_of_week'] = df.index.day_of_week
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    return df

def iforest_predict_proba(df,**kwargs):
    iforest = IForest(**kwargs).fit(df.fillna(0))
    probs = iforest.predict_proba(df.fillna(0))[:,1]
    return probs

def probs_threshold(probs,threshold):
    is_outlier = probs > threshold
    return is_outlier

def isolation_forest_pred(df,threshold=0.75,**kwargs):
    df = add_date_features(df)
    probs = iforest_predict_proba(df,**kwargs)
    is_outlier = probs_threshold(probs,threshold)
    return probs, is_outlier

# Scale *all* features (for now) with Quantile Transformer to normal distribution
def quantile_transform(df):
    df = df.copy()
    
    qt = QuantileTransformer(output_distribution="normal")
    
    df.loc[:,:] = qt.fit_transform(df)
    
    return qt, df

estimators = [KNN(n_neighbors=20), LOF(n_neighbors=20), IForest()]

def train_ensemble(estimators,df,threshold=0.75):
    
    df = df.copy().fillna(0)
    
    # Create an empty array
    shape = (len(df), len(estimators))
    probability_scores = np.empty(shape=shape)

    # Loop over and fit
    for index, est in enumerate(estimators):
        est.fit(df)

        # Create probabilities    
        probs = est.predict_proba(df)

        # Store the probs    
        probability_scores[:, index] = probs[:, 1]

    mean_scores = np.mean(probability_scores, axis=1)
    mean_scores    

    median_scores = np.mean(probability_scores, axis=1)
    median_scores

    # Create a mask with 75% threshold
    is_outlier = median_scores > threshold

    # Filter the outliers
    # outliers = df[is_outlier]
    
    return is_outlier, mean_scores, median_scores






















header = 'Детекция аномалий по признакам, которые могут быть связаны с уходом от налогов/мошенническими действиями.'

subheader = '''
Исследуем характеристики ввода, транспортировки и вывода конкретного товара (`gtin`), конкретного владельца карточки (`prid`) в поисках аномалий, которые могли бы указывать на мошеннические действия.
'''
#################
application_fields = '''### Для кого: 

* **налоговых органов** 
* **органов рыночного надзора**'''

economy_benefit = '''### Польза для экономики:

* Повышение налоговых отчислений
* Детекция новых схем ухода от налогов
* Повышение рыночной прозрачности

'''

consumer_benefit = '''### Польза для потребителей:

* Увеличение доли настоящей продукции
* Контроль за необоснованным ростом цен
* Большая уверенность в товарной истории и производителе
'''
#################
features_short = '''1. **Накопление невыведенных товаров**
2. **Аномальная цена продажи**
3. **Аномальное кол-во перемещений**
4. **Аномальное кол-во участников перемещений**
5. **Аномальное качество перемещений**
6. **Аномально высокий объем вывода**'''
#################
concept = '''**Выделяем 6 признаков, которые гипотетически могут указывать на мошеннические действия такого рода. Предполагаем два подхода к анализу этих признаков:**
1. Каждый признак проверяем на выбросы (аномалии) отдельно и считаем кол-во аномальных признаков. Чем больше, тем больше шансов, что это True Positive
2. И совокупно с помощью ML, предполагая детекцию сложных связей.



**6 Признаков:**

1. **Накопление невыведенных товаров для конкретных GTIN и конкретных ИНН**

    * Если (`ввод_товаров` - `вывод_товаров`) имеет тенденцию к росту, или происходят резкие скачки, которые не имеют тенденцию снижаться, то это может быть свидетельством:

    * За мошенничество: попытка скрыть вывод товара, снизить налоговую базу и подобное

    * Против мошенничества: проблемы с системой отслеживания на каком то этапе, товар, который "долго продается"

    * Анализируем временной ряд таких накоплений товаров. Ищем резкие необоснованные скачки, либо медленное размытое увеличение.

2. **Цена продажи GTIN для конкретного ИНН сильно изменилась (переключение режима?)**

    * Цена сильно изменилась от ожидаемой как в большую так и в меньшую сторону.

    * За: Уменьшение цены - использование нелегальных, дешевых материалов, несоблюдение стандартов. Если цена изменяется необоснованно, это вызывает вопрос. Увеличение цены - потенциальное использование сложных схем или перекладывание рисков между участниками системы.

    * Против: Изменился вес, объем, характеристики товара, др. (В этом случае будет ли перерегистрироваться GTIN?)

3. **Количество перемещений товара (GTIN) до реализации превышает ожидаемое этого же товара, этого же ИНН**

    * За: мошеннические схемы, "карусель спроса", уход от налогов

    * Против: форс-мажор, перестройка схемы доставки

    * Альтернатива: сравнивать со средним по GTIN со всеми ИНН в одном регионе.

4. **Кол-во участников в перемещении товара (GTIN) с конкретным ИНН превышает ожидаемое**

    * За: мошеннические схемы, "карусель спроса", уход от налогов

    * Против: форс-мажор, перестройка схемы доставки

5. **Необоснованное изменение качества перемещения**

    * Пока не ясно, как считать. Например овощи везут в другой регион, хотя все время продавались в регионе производства.

    * За: Попытка скрыть происхождение товара?

    * Против: Новые каналы сбыта?

6. **Аномально высокий объем вывода товара конкретного GTIN, конкретного ИНН или в разрезах только GTIN и только ИНН**

    * За: продавец может создавать ложные продажи, чтобы нарушить правила программы лояльности или увеличить свою прибыль за счет схем, связанных с возвратом товаров или снижением цен, которые не соответствуют их качеству

    * Против: Сезонность, изменение бизнес модели
'''

#################
preprocessing_description = '''1. Вытаскиваем из 3х таблиц все записи с выбранной парой `prid`-`gtin`
2. Ресэмплим периодом в 1 день
3. Считаем: 
    * сумму кол-ва ввода в оборот, 
    * кол-во перемещений, 
    * кол-во отправителей, 
    * кол-во получаетелей, 
    * среднее, медиану и сумму кол-ва перемещенных товаров
    * среднее и медиану цены вывода
    * сумму вывода
4. Заполняем пропуски 0
5. Обогащаем признаками:
    * Кумулятивная сумма ввода
    * Кумулятивная сумма выводы
    * Разница между кумулятивными суммами выше
    * Дифференцированный признака выше (для стационарности)
    * Долю получателей/отправителей
    * Сумму получателей и отправителей
6. Заполняем пропуски 0


'''

preprocessing_code = '''
# ф-ция "вытаскивания" данных о конкретном товаре и производителе из всех данных
def slice_and_transform(inn_prid,gtin,df,df1,df2, fillna=True):
    
    # in table
    temp_df_in = df.loc[(df['prid'] == inn_prid)&(df['gtin'] == gtin),['dt','cnt']]
    temp_df_in.columns = ['dt','cnt_in']
    temp_df_in['dt'] = pd.to_datetime(temp_df_in['dt'])
    temp_df_in = temp_df_in.set_index('dt')
    temp_df_in = temp_df_in.sort_index()
    temp_df_in = temp_df_in.resample('1D').sum()
    
    # move table
    temp_df_mv = df1.loc[(df1['prid'] == inn_prid)&(df1['gtin'] == gtin)] 
    temp_df_mv['dt'] = pd.to_datetime(temp_df_mv['dt'])
    temp_df_mv = temp_df_mv.set_index('dt')
    temp_df_mv = temp_df_mv.sort_index()
    resample_dict = {'gtin':'count','sender_inn':'nunique','receiver_inn':'nunique','cnt_moved':['mean','median','sum']}
    temp_df_mv = temp_df_mv.resample('1D').agg(resample_dict)
    temp_df_mv.columns = ['count','sender_inn_nunique','receiver_inn_nunique','cnt_moved_mean','cnt_moved_median','cnt_moved_sum']
    
    # out table
    temp_df_out = df2.loc[(df2['prid'] == inn_prid)&(df2['gtin'] == gtin),['dt','price','cnt']] 
    temp_df_out.columns = ['dt','price_out','cnt_out']
    temp_df_out['dt'] = pd.to_datetime(temp_df_out['dt'])
    temp_df_out = temp_df_out.set_index('dt')
    temp_df_out = temp_df_out.sort_index()
    temp_df_out = temp_df_out.resample('1D').agg({'price_out':['mean','median'],'cnt_out':'sum'})
    temp_df_out.columns = ['price_out_mean','price_out_median','cnt_out_sum']
    
    if fillna:
        temp_df_in = temp_df_in.fillna(0)
        temp_df_mv = temp_df_mv.fillna(0)
        temp_df_out = temp_df_out.fillna(0)
    
    temp_df = temp_df_in.join(temp_df_out,how='outer').join(temp_df_mv,how='outer')
    
    
    return temp_df


# Ф-ция добавления признаков
def enrich(df):
#     Create in out cumsum delta
    df['cnt_in_cumsum'] = df['cnt_in'].cumsum()
    df['cnt_out_cumsum'] = df['cnt_out_sum'].cumsum()
    df['in_out_delta'] = df['cnt_in_cumsum'] - df['cnt_out_cumsum']
    df['in_out_delta_stationary'] = df['in_out_delta'].diff()
    
#     Create sender_receiver_ratio
    df['sender_receiver_ratio'] = df['sender_inn_nunique'] / df['receiver_inn_nunique']
    df['sender_receiver_sum'] = df['sender_inn_nunique'] + df['receiver_inn_nunique']
    
    
    # df = df.fillna(0)
    
    return df


def preprocess(inn_prid,gtin,df,df1,df2, fillna=True):
    result = slice_and_transform(inn_prid,gtin,df,df1,df2, fillna=True)
    result = enrich(result)
    return result
'''
#################
mad_description = '''### MAD (Median Absolute Deviation)
```
statsmodels.tsa.seasonal.seasonal_decompose
pyod.models.mad.MAD
```
Метод детекции аномалий одного признака, не требует нормального распределения.

Детекция аномалии на остатках сезонной декомпозиции признака позвляет учесть влияние трендового и сезонного компонента временного ряда.

В целом метод помечает в строках признаки с аномалиями и, исходя из кол-ва таких аномальных признаков, помечает строку как требующую внимания.
Легко интерпретировать, потому что конкретно понятно, где именно аномалия.

'''

mad_code = '''def detect_outliers_on_residuals_with_mad(df,cols=None):
    
    def mad_on_residuals(df,col_name):
        resid = seasonal_decompose(df[col_name].fillna(0)).resid.values.reshape(-1, 1)
        mad = MAD().fit(resid)
        is_outlier = mad.labels_ == 1
        # display(mad.threshold_)
        # display(mad.decision_scores_)
        return is_outlier
    
    if cols==None:
        cols = ['in_out_delta_stationary','price_out_mean', 'price_out_median',
                'cnt_moved_mean','cnt_moved_median', 'cnt_moved_sum','cnt_out_sum']
        

    outliers = pd.DataFrame(columns=cols)


    for col in cols:
        is_outlier = mad_on_residuals(df,col)
        outliers[col] = is_outlier
    
    cols = [x+'_resid_mad' for x in cols]
    outliers.columns = cols
    
    outliers['sum_resid_mad'] = outliers.sum(axis=1)
    
    return outliers
'''
#################
iforest_description = '''```
pyod.models.iforest.IForest
```
Модели машинного обучения позволяют обнаружить аномалии, как сочетание признаков. (multivariate)

Достоинства:
* эффективен на больших датасетах
* не требует нормального распределения признаков
* не опирается ни на какие статистические предположения
* хорошо работает из коробки

В предобработке добавляем признаки из даты.

В целом подход может использоваться отдельно или в комбинации с MAD на остатках.
'''

iforest_code = '''def add_date_features(df):
    df = df.copy()
    df['day_of_week'] = df.index.day_of_week
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    return df

def iforest_predict_proba(df,**kwargs):
    iforest = IForest(**kwargs).fit(df.fillna(0))
    probs = iforest.predict_proba(df.fillna(0))[:,1]
    return probs

def probs_threshold(probs,threshold):
    is_outlier = probs > threshold
    return is_outlier

def isolation_forest_pred(df,threshold=0.75,**kwargs):
    df = add_date_features(df)
    probs = iforest_predict_proba(df,**kwargs)
    is_outlier = probs_threshold(probs,threshold)
    return probs, is_outlier
'''

#################
ensemble_description = '''```
sklearn.preprocessing.QuantileTransformer
pyod.models.iforest.IForest
pyod.models.knn.KNN
pyod.models.lof.LOF
```
Композиции моделей дают более стабильные прогнозы. Их тоже можно использовать в сочетании с univariate методами, типа MAD на остатках.

Некотоыре модели в этой композии предполагают нормальное распределение признаков, поэтому применяем Quantile Transformer, который способен трансформировать распределение в нормальное.
'''
ensemble_code = '''# Scale *all* features (for now) with Quantile Transformer to normal distribution
def quantile_transform(df):
    df = df.copy()
    
    qt = QuantileTransformer(output_distribution="normal")
    
    df.loc[:,:] = qt.fit_transform(df)
    
    return qt, df

estimators = [KNN(n_neighbors=20), LOF(n_neighbors=20), IForest()]

def train_ensemble(estimators,df,threshold=0.75):
    
    df = df.copy().fillna(0)
    
    # Create an empty array
    shape = (len(df), len(estimators))
    probability_scores = np.empty(shape=shape)

    # Loop over and fit
    for index, est in enumerate(estimators):
        est.fit(df)

        # Create probabilities    
        probs = est.predict_proba(df)

        # Store the probs    
        probability_scores[:, index] = probs[:, 1]

    mean_scores = np.mean(probability_scores, axis=1)
    mean_scores    

    median_scores = np.mean(probability_scores, axis=1)
    median_scores

    # Create a mask with 75% threshold
    is_outlier = median_scores > threshold

    # Filter the outliers
    # outliers = df[is_outlier]
    
    return is_outlier, mean_scores, median_scores
'''


#################
scalability = '''### Масштабируемость

1. Предобработка может происходить целиком в SQL запросе
2. Модели в целом легкие, распараллеливаемые
'''

improvements = '''### Улучшение

#### Исследование доменной области:

* Можно глубже изучить различные мошеннические схемы, выделить типовые признаки и использовать их тоже.
* Также крайне эффективным кажется получение обратной связи от аналитика соответствующего гос. органа для формирования гипотез, генерации признаков, более точной настройки моделей и выбора технологий в целом.
* Создание размеченных данных для обучения с учителем (такие методы предпочтительнее обучению без учителя)

#### Другие технологии:

* Мы здесь в основном использовали библиотеку PyOD, но в документации для детекции аномалий на временных рядах рекомендуется использовать библиотеку TODS: https://github.com/datamllab/tods

#### Другие подходы:

 * Аномалии могут детектироваться не только по конкретным записям (**time points as outliers**), но и по последовательностям (**pattern-wise detection - subsequences as outliers**) и целиком по временным рядам (**system-wise detection - sets of time series as outliers**)
 * Можно создать какую-то совокупную оценку владельца `gtin`. Например, как много у него конкретно аномальных записей или другое.

'''

ui_ux = '''#### UI/UX

* Можно добавить больше инфы про конкретный `prid`-`gtin`, в целом по `gtin`, в целом по ИНН.
* Можно упростить вывод инфы о детекции аномалий. Сделать композитное решение, которое будет выводить просто строку и уверенность модели.
'''
