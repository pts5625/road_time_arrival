'''
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_squared_error

def read_dataframe(filename):
    df = pd.read_csv(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

df_train = read_dataframe('./data/green_tripdata_2025-01.csv')
df_val = read_dataframe('./data/green_tripdata_2024-06.csv')

df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

mean_squared_error(y_val, y_pred, squared=False)

with mlflow.start_run():

    mlflow.set_tag("developer", "cristian")

    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")

    alpha = 0.1
    mlflow.log_param("alpha", alpha)
    lr = Lasso(alpha)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")

'''
import pandas as pd
df= pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-02.parquet')
print(df.head())

df['duration']=df['tpep_dropoff_datetime']-df['tpep_pickup_datetime']
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
df=df.loc[((df.duration > 25) & (df.duration <50)) & ((df.trip_distance >4) & (df.trip_distance <14))]
df.dropna(inplace=True)

from sklearn.model_selection import train_test_split

# imprt elasticnet
from sklearn.linear_model import ElasticNet
parameters= {
    'alpha': 0.1,
    'l1_ratio': 0.5
}
model= ElasticNet(
    alpha=parameters['alpha'],
    l1_ratio=parameters['l1_ratio']
)

X = df[['passenger_count', 'trip_distance','duration']]
y = df['total_amount']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
from sklearn.metrics import root_mean_squared_error, r2_score
r2_scores = r2_score(y_val, y_pred)

rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE: {rmse}")

from sklearn.ensemble import RandomForestRegressor

rf_parameters = {
    'n_estimators': 100,
    'random_state': 42
}
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_rmse = root_mean_squared_error(y_val, rf_pred)
rf_r2_score = r2_score(y_val, rf_pred)

import dagshub
import mlflow
dagshub.init(repo_owner='pts5625', repo_name='road_time_arrival', mlflow=True)

mlflow.set_experiment('road_time_arrival')
mlflow.set_tracking_uri('https://dagshub.com/pts5625/road_time_arrival.mlflow')


with mlflow.start_run():

    mlflow.set_tag("developer", "pts5625")
    
    mlflow.log_param("alpha", parameters['alpha'])
    mlflow.log_param("l1_ratio", parameters['l1_ratio'])
    mlflow.log_param("n_estimators", rf_parameters['n_estimators'])
    mlflow.log_param("random_state", rf_parameters['random_state'])
    
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2_scores)
    mlflow.log_metric("rf_rmse", rf_rmse)
    mlflow.log_metric("rf_r2_score", rf_r2_score)

    mlflow.sklearn.log_model(model, "model")

