import os
import yaml
import pandas as pd
import datetime as dt

dir_ = os.path.join(os.getcwd(), 'config.yaml')
with open(fr'{dir_}', 'r') as file:
    CONFIG_ = yaml.safe_load(file)


class Dataset:
    def __init__(self):
        self.dataset = self.load_data(CONFIG_['loading_dataset_path'])
        self.create_columns()
        self.drop_columns()
        self.save_data(CONFIG_['saving_dataset_path'])
        
    def load_data(self, path):
        df = pd.read_csv(path)
        return df
    
    def save_data(self, path):
        self.dataset.to_csv(path)
        return
    
    def create_columns(self):
        self.convert_date()
        self.get_month()
        self.get_quarter()
        self.get_year()
        self.get_postal_district()
        self.get_storey_type()
        return

    def convert_date(self):
        self.dataset['month'] = pd.to_datetime(self.dataset['month'], infer_datetime_format=True)
        return
    
    def get_postal_district(self):
        self.dataset['postal_district'] = self.dataset['postal_code'].str.slice(0,2)
        return
    
    def get_year(self):
        self.dataset['year'] = self.dataset['month'].dt.year
        return
    
    def get_quarter(self):
        self.dataset['quarter_period'] = self.dataset['month'].dt.quarter
        return
    
    def get_month(self):
        self.dataset['month_period'] = self.dataset['month'].dt.month
        return
    
    @staticmethod
    def group_storey_range(storey_range):
        if storey_range in ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12']:
            return storey_range
        else:
            return '13+'
    
    def get_storey_type(self):
        self.dataset['storey_range_grouped'] = self.dataset['storey_range'].apply(
            Dataset.group_storey_range)
        return
    
    def drop_columns(self):
        self.dataset.drop(columns=['Unnamed: 0',
                                   'flat_model',
                                   'street_name',
                                   'remaining_lease',
                                   'lease_commence_date',
                                   '_id',
                                   'block',
                                   'psm',
                                   'address',
                                   'latitude',
                                   'longitude',
                                   'postal_code',
                                   'mrt',
                                   'stn_no'],
                          inplace=True)
        return
