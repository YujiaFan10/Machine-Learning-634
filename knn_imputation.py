# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:29:50 2019

@author: Karra's
"""
from __future__ import absolute_import, print_function, division
import sys
import numpy as np
import pandas as pd
from sklearn import neighbors

from scipy.spatial import distance
from scipy import stats

##########1st type k we can choose
def euc(a, b):
    return distance.euclidean(a,b)
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.k_closest(row, 5) # change the number here to increase the number of neighbours
            predictions.append(label)
        return np.asarray(predictions)

    
    def k_closest(self, row, k):
        knn_label = []
        neighbor_dist = []
        for i in self.X_train:
            dist = euc(row, i)
            neighbor_dist.append(dist)  # compute the distance from all neighbors
        ndist = np.array(neighbor_dist) # convert the list of distance into an array
        knn = ndist.argsort()[:k]  # find the index of the k closest values
        for j in knn:
            knn_label.append(self.y_train[j])  # categorising
        pred = stats.mode(knn_label)[0][0]  # finding the most frequently occured values
        return pred.astype(int) #convert the value back to int as mode
    
#############k =1 fixed################
class KNN():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = distance.euclidean(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = distance.euclidean(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]

##########for 1 nearest neighbour
class KNNClassifier(object):
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def euc_distance(self, a, b):
        return np.linalg.norm(a-b)
        # return distance.euclidean(a, b)

    def closest(self, row):
        """
        Returns the label corresponding to the single closest training example.
        This is a k=1 nearest neighbor(s) implementation.
        :param row:
        :return:
        """
        dist = [self.euc_distance(row, trainer) for trainer in self.X_train]
        best_index = dist.index(min(dist))

        return self.y_train[best_index]

    def fit(self, training_data, training_labels):
        self.X_train = training_data
        self.y_train = training_labels

    def predict(self, to_classify):
        predictions = []
        for row in to_classify:
            label = self.closest(row)
            predictions.append(label)

        return predictions
##########################3
class Imputer:
    """Imputer class."""

    def _fit(self, X, column, k=10, is_categorical=False):
        """Fit a knn classifier for missing column.
        - Args:
                X(numpy.ndarray): input data
                column(int): column id to be imputed
                k(int): number of nearest neighbors, default 10
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                clf: trained k nearest neighbour classifier
        """
        clf = None
        if not is_categorical:
            clf = neighbors.KNeighborsRegressor(n_neighbors=k)
        else:
            #clf = neighbors.KNeighborsClassifier(n_neighbors=k)
            clf = ScrappyKNN()
        # use column not null to train the kNN classifier
        missing_idxes = np.where(pd.isnull(X[:, column]))[0]
        if len(missing_idxes) == 0:
            return None
        X_copy = np.delete(X, missing_idxes, 0)
        X_train = np.delete(X_copy, column, 1)
        # if other columns still have missing values fill with mean
        col_mean = None
        if not is_categorical:
            col_mean = np.nanmean(X, 0)
        else:
            col_mean = np.nanmedian(X, 0)
        for col_id in range(0, len(col_mean) - 1):
            col_missing_idxes = np.where(np.isnan(X_train[:, col_id]))[0]
            if len(col_missing_idxes) == 0:
                continue
            else:
                X_train[col_missing_idxes, col_id] = col_mean[col_id]
        y_train = X_copy[:, column]
        # fit classifier
        clf.fit(X_train, y_train)
        return clf

    def _transform(self, X, column, clf, is_categorical):
        """Impute missing values.
        - Args:
                X(numpy.ndarray): input numpy ndarray
                column(int): index of column to be imputed
                clf: pretrained classifier
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                X(pandas.dataframe): imputed dataframe
        """
        missing_idxes = np.where(np.isnan(X[:, column]))[0]
        X_test = X[missing_idxes, :]
        X_test = np.delete(X_test, column, 1)
        # if other columns still have missing values fill with mean
        col_mean = None
        if not is_categorical:
            col_mean = np.nanmean(X, 0)
        else:
            col_mean = np.nanmedian(X, 0)
        # fill missing values in each column with current col_mean
        for col_id in range(0, len(col_mean) - 1):
            col_missing_idxes = np.where(np.isnan(X_test[:, col_id]))[0]
            # if no missing values for current column
            if len(col_missing_idxes) == 0:
                continue
            else:
                X_test[col_missing_idxes, col_id] = col_mean[col_id]
        # predict missing values
        y_test = clf.predict(X_test)
        X[missing_idxes, column] = y_test
        return X

    def knn(self, X, column, k=10, is_categorical=False):
        """Impute missing value with knn.
        - Args:
                X(pandas.dataframe): dataframe
                column(str): column name to be imputed
                k(int): number of nearest neighbors, default 10
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                X_imputed(pandas.dataframe): imputed pandas dataframe
        """
        X, column = self._check_X_y(X, column)
        clf = self._fit(X, column, k, is_categorical)
        if clf is None:
            return X
        else:
            X_imputed = self._transform(X, column, clf, is_categorical)
            return X_imputed

    def _check_X_y(self, X, column):
        """Check input, if pandas.dataframe, transform to numpy array.
        - Args:
                X(ndarray/pandas.dataframe): input instances
                column(str/int): column index or column name
        - Returns:
                X(ndarray): input instances
        """
        column_idx = None
        if isinstance(X, pd.core.frame.DataFrame):
            if isinstance(column, str):
                # get index of current column
                column_idx = X.columns.get_loc(column)
            else:
                column_idx = column
            X = X.as_matrix()
        else:
            column_idx = column
        return X, column_idx
    
#from imputer import Imputer
        




#######################################    
impute = Imputer()    
#X=[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[3,4,np.nan],[4,np.nan,5]]
#g=pd.DataFrame(X,columns=["a","b","c"])

imputed_dataframe=cat_data_new.copy()
k=0
#for col in cat_data.columns:
 #   cat_data[col] = cat_data[col].astype('category')
cat_data_new=cat_data.astype(np.float64)

for i in cat_data_new.columns:
    
    X_imputed_knn = impute.knn(X=cat_data_new,column =i,is_categorical=True)
    imputed_dataframe.loc[:,i]=X_imputed_knn[:,k]
    k+=1
    print(k)