# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:43:33 2019

@author: Karra's
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:03:30 2019

@author: Karra's
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:55:16 2019

@author: Karra's
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from tqdm import tqdm
import lightgbm as lgb
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier




class BaseMICE:
    """Base class for the MICE implementation."""
    
    def __init__(self, max_iter=10):
        self.max_iter = max_iter
    
    def fill_missing_values(self, df):
        """Fills the missing values of a pandas DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input data with missing values (nans).
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with imputed missing values.
        """
        nan_ids = np.argwhere(df.isna().values).tolist()
        df_imputed = self.impute_initial_mean_or_mode(df)
        iter_results = []
        for iter in range(self.max_iter):
            df_imputed = self.transform(df_imputed, nan_ids, iter)
        return df_imputed
    def fill_missing_values_and_result(self, df):
        """Fills the missing values of a pandas DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input data with missing values (nans).
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with imputed missing values.
        """
        nan_ids = np.argwhere(df.isna().values).tolist()
        df_imputed = self.impute_initial_mean_or_mode(df)
        iter_results = []

        return df_imputed    
    
    def benchmark(self, df_missing, drop_columns_loss=None):
        """Benchmarks the fill method for missing values.
        
        Parameters
        ----------
        df_original : pandas.DataFrame
            Original data.
        df_missing : pandas.DataFrame
            Input data with missing values (nans).
        drop_columns_loss : list, optional
            Drop columns in the result DataFrame when 
            computing the loss.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with imputed missing values.
        """
        columns_missing = df_missing.isna().sum()
        columns_missing = columns_missing[columns_missing > 0]
        nan_ids = np.argwhere(df_missing.isna().values).tolist()
        df_imputed = self.impute_initial_mean_or_mode(df_missing)
      #  df_missing=df_missing.replace(np.nan,-999)
      #  df_missing=df_missing.astype(int)
        
        self.df_mean = df_imputed.copy()
        
        iter_results = []
        for iter in range(self.max_iter):
            time_start = time()
            df_imputed = self.transform(df_imputed, columns_missing, nan_ids, iter)
            time_stop = time() - time_start
           # if drop_columns_loss:
            #    loss = self.compute_loss(df_original.drop(columns=drop_columns_loss, axis=1), 
             #                            df_imputed.drop(columns=drop_columns_loss, axis=1))
           # else:
              #  loss = self.compute_loss(df_original, df_imputed)
            iter_results.append({
                "iter": iter,
                "time_seconds": time_stop, 
                #"loss": loss
            })
        return df_imputed,iter_results
    
    def benchmark_mean_loss(self, df_original, df_missing, drop_columns_loss=None):
        """Computes the same iterations as benchmark() but only for the of the mean imputation method."""
        time_start = time()
        df_imputed = self.impute_initial_mean_or_mode(df_missing)
        time_stop = time() - time_start
        
        iter_results = []
        for iter in range(self.max_iter):
            if drop_columns_loss:
                loss = self.compute_loss(df_original.drop(columns=drop_columns_loss, axis=1), 
                                         df_imputed.drop(columns=drop_columns_loss, axis=1))
            else:
                loss = self.compute_loss(df_original, df_imputed)
            iter_results.append({
                "iter": iter,
                "time_seconds": time_stop, 
                "loss": loss
            })
        return iter_results
    
    def get_model(self, target):
        if is_numeric_dtype(target):
            model = DecisionTreeRegressor()
        else:
          #  model =decision_tree(max_depth=5,min_num_split=2)
            model = DecisionTreeClassifier()
        return model
    
    def compute_loss(self, original_df, filled_df):
        """Computes the difference between the original and filled DataFrames."""
        return mean_squared_error(original_df, filled_df)
    
    def impute_initial_mean_or_mode(self, df):
        df_new = df.copy()
        for column in df:
            if is_numeric_dtype(df[column]):
                df_new[column] = df_new[column].fillna(df_new[column].mean()).astype(int)
            else:
                df_new[column] = df_new[column].fillna(df_new[column].mode()[0])
               # df_new[column] = df_new[column].fillna(-1)
        return df_new
    
    def transform(self, df, nan_ids,it):
        pass
 

class FastMICE(BaseMICE):
    """MICE implementation using column by column imputation."""
    
    method_name = "Fast MICE"
    
    def transform(self, df: pd.DataFrame, columns_missing: list, nan_ids: list, iter_id: int):
        #rand_column_ids = range(len(columns_missing)).tolist()
        for col in df.columns:
            df[col] = df[col].astype('category')
        rand_column_ids =[x for x in range(len(columns_missing))]
        for column_id in tqdm(rand_column_ids, desc="{method_name}: Iter {ie} / {max_iter}".format(method_name=self.method_name,ie=iter_id,max_iter=self.max_iter), position=0):
            target_column_name = columns_missing.index[column_id]
            X = df.drop(columns=[target_column_name], axis=1)
          #  X = pd.get_dummies(X, drop_first=True)
            y = df[target_column_name]
            n=df.columns.get_loc(target_column_name)
            column_nan_ids = [id[0] for id in nan_ids if id[1] ==n]
            
            # Fit model
            #for col in X.columns:
             #   X[col] = X[col].astype('category')
            y=y.astype('category')
            model = self.get_model(y).fit(X.drop(index=column_nan_ids), y.drop(index=column_nan_ids))
            #model = self.get_model(y).fit(df,target_column_name)
            # Predict value
            df.iloc[column_nan_ids, n] = model.predict(X.iloc[column_nan_ids, :])
        return df

class VanilaMICE(BaseMICE):
    """MICE implementation using value by value imputation."""
    
    method_name = "Vanila MICE"
    
    def transform(self, df: pd.DataFrame, columns_missing: list, nan_ids: list, iter_id: int):
        #random_ids = np.random.permutation(len(nan_ids)).tolist()
        for col in df.columns:
                df[col] = df[col].astype('category')
        for id in tqdm(columns_missing, desc="{method_name}: Iter {ie} / {max_iter}".format(method_name=self.method_name,ie=iter_id,max_iter=self.max_iter), position=0):
            # Setup data
            row_id, col_id = nan_ids[id]
            target_column_name = df.columns[col_id]
            X = df.drop(columns=[target_column_name], axis=1)
            #X = pd.get_dummies(X, drop_first=True)
            y = df[target_column_name]
            
            # Fit model

            y=y.astype('category')
            model = self.get_model(y).fit(X.drop(index=row_id), y.drop(index=row_id))
            
            # Predict value
            df.iloc[row_id, col_id] = model.predict(X.iloc[row_id:row_id + 1, :])
        return df

class SlowFastMICE(BaseMICE):
    """MICE implementation using value-by-value imputation in the first iteration 
    and column-by-column for the remaining iterations."""
    
    method_name = "Slow-Fast MICE"
    
    def __init__(self, max_iter=10):
        self.max_iter = max_iter
        self.vanila_mice = VanilaMICE(self.max_iter)
        self.vanila_mice.method_name = self.method_name
        self.fast_mice = FastMICE(self.max_iter)
        self.fast_mice.method_name = self.method_name
    
    def transform(self, df: pd.DataFrame, columns_missing: list, nan_ids: list, iter_id: int):
        if iter_id > 0:
            return self.fast_mice.transform(df, columns_missing, nan_ids, iter_id)
        else:
            return self.vanila_mice.transform(df, columns_missing, nan_ids, iter_id)


class FastSlowMICE(BaseMICE):
    """MICE implementation using column-by-column imputation in all but one iterations
    and value-by-value for the last iteration."""
    
    method_name = "Fast-Slow MICE"
    
    def __init__(self, max_iter=10):
        self.max_iter = max_iter
        self.vanila_mice = VanilaMICE(self.max_iter)
        self.vanila_mice.method_name = self.method_name
        self.fast_mice = FastMICE(self.max_iter)
        self.fast_mice.method_name = self.method_name
    
    def transform(self, df: pd.DataFrame, columns_missing: list, nan_ids: list, iter_id: int):
        if iter_id + 1 == self.max_iter:
            return self.vanila_mice.transform(df, columns_missing, nan_ids, iter_id)
        else:
            return self.fast_mice.transform(df, columns_missing, nan_ids, iter_id)
        
new=BaseMICE(10).fill_missing_values_and_result(cat_data)
new2=FastMICE(10).benchmark(cat_data)
new3=VanilaMICE(10).benchmark(cat_data)
new4=SlowFastMICE(10).benchmark(cat_data)
new5=FastSlowMICE(10).benchmark(cat_data)