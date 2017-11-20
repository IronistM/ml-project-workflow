"""
custom_transformers.py
~~~~~~~~~~~~~~~~~~~~~~

This module contains custom Scikit-Learn transfomers for use
in the data preparation pipeline used in this data science 
case study.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class DataFrameAdapter(BaseEstimator, TransformerMixin):
    """DataFrameAdapter
    
    Class for mapping column-subsets of Pandas DataFrames
    to raw Numpy arrays.
    """
    def __init__(self, col_names):
        self.col_names = list(col_names)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.col_names].values
    
    def get_feature_names(self):
        return self.col_names


class CategoricalFeatureEncoder(BaseEstimator, TransformerMixin):
    """CategoricalFeatureEncoder
    
    Class for automating the process of applying 
    one-hot-encoding to all categorical variables in a Numpy
    array of only categorical variables.
    """
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        num_vars = X.shape[1]
        encoded_vars = [self.__transform_single_cat_var__(var) for var in X.T]
        feature_names, features = list(zip(*encoded_vars))
        self.feature_names = list(np.concatenate(feature_names, axis=0))
        return np.concatenate(features, axis=1)

    def get_feature_names(self):
        return self.feature_names
    
    def __transform_single_cat_var__(self, cat_feature_col):
        feature_names_, int_factors = np.unique(cat_feature_col, return_inverse=True)
        one_hot_encoder = OneHotEncoder()
        encoded_factors = one_hot_encoder.fit_transform(int_factors.reshape((-1, 1)))
        return (feature_names_, encoded_factors.toarray())
