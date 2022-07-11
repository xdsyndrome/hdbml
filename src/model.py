import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
import sys

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    """_summary_

    Args:
        actual (_type_): _description_
        pred (_type_): _description_

    Returns:
        _type_: rmse, mae, r2
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def preprocessing(dataset):
    dataset.drop(columns=['town',
                          'month',
                          'storey_range',
                          'Unnamed: 0'],
                 inplace=True)
    
    numeric_features = ['floor_area_sqm',
                        'resale_price',
                        'remaining_lease_years',
                        'distance_meters']
    dataset['month_period'] = dataset['month_period'].astype(str)
    dataset['quarter_period'] = dataset['quarter_period'].astype(str)
    dataset['year'] = dataset['year'].astype(str)
    categorical_features = [x for x in dataset.columns if x not in numeric_features]
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_features)]
    )
    clf = Pipeline(steps=[('preprocessor', preprocessor)])
    return clf


def run_model():
    np.random.seed(123)
    try:
        dataset = pd.read_csv('data/dataset_model.csv')
    except Exception as e:
        logger.exception(
            'Unable to open dataset. Error: %s', e
        )
    
    clf = preprocessing(dataset)
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop('resale_price', axis=1),
                                                        dataset['resale_price'],
                                                        train_size=0.8)
    xgbr = XGBRegressor()
    clf.steps.append(['regressor', xgbr])
    
    with mlflow.start_run():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, pred)
        mlflow.log_metrics({'rmse': rmse,
                            'mae': mae,
                            'r2': r2})
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(xgbr, "model", registered_model_name="XGBoostHDB")
        else:
            mlflow.sklearn.log_model(xgbr, "model")
    
run_model()