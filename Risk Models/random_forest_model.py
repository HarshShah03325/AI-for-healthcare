import itertools
import pandas as pd

from IPython.display import Image 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from random_forest_utils import load_data, cindex, random_forest_grid_search, hyperparams, holdout_grid_search

class Random_Forest_Model():
    def __init__(self):
        X_dev, X_test, y_dev, y_test = load_data(10)
        self.X_dev = X_dev
        self.X_test = X_test
        self.y_dev = y_dev
        self.y_test = y_test

    
    def drop_missing_values(self):
        """
        Function to calculate c-index and best hyperparamters for given dataset.
        Drop the missing values(NaN) and compute a random forest classifier.

        returns:
        c-index of the dataset of patients by dropping the missing values.
        """
        X_train, X_val, y_train, y_val = train_test_split(self.X_dev, self.y_dev, test_size=0.25, random_state=10)
        y_test = self.y_test
        X_test = self.X_test
        X_train_dropped = X_train.dropna(axis='rows')
        y_train_dropped = y_train.loc[X_train_dropped.index]
        X_val_dropped = X_val.dropna(axis='rows')
        y_val_dropped = y_val.loc[X_val_dropped.index]
        X_test_dropped = X_test.dropna(axis='rows')
        y_test_dropped = y_test.loc[X_test_dropped.index]
        
        best_rf, best_hyperparams = random_forest_grid_search(X_train_dropped, y_train_dropped, X_val_dropped, y_val_dropped)
        y_train_best = best_rf.predict_proba(X_train_dropped)[:, 1]
        Train_c_index = cindex(y_train_dropped, y_train_best)
        
        y_val_best = best_rf.predict_proba(X_val_dropped)[:, 1]
        valid_c_index = cindex(y_val_dropped, y_val_best)
        
        y_test_best = best_rf.predict_proba(X_test_dropped)[:,1]
        test_c_index = cindex(y_test_dropped, y_test_best)
        
        return Train_c_index, valid_c_index, test_c_index
    

    def mean_impute(self):
        """
        Function to compute c-index and best hyperparameters for given dataset.
        Replace all the missing values(NaN) with the mean of all available values and train a random forest classifier.

        returns:
        c-index of the dataset of patients calculated after using mean imputation.
        """
        X_train, X_val, y_train, y_val = train_test_split(self.X_dev, self.y_dev, test_size=0.25, random_state=10)
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(X_train)
        X_train_mean_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
        X_val_mean_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
        rf = RandomForestClassifier
        rf_mean_imputed, best_hyperparams_mean_imputed = holdout_grid_search(rf, X_train_mean_imputed, y_train,
                                                                     X_val_mean_imputed, y_val,
                                                                     hyperparams, {'random_state': 10})

        y_train_best = rf_mean_imputed.predict_proba(X_train_mean_imputed)[:, 1]
        train_c_index = cindex(y_train, y_train_best)
       
        y_val_best = rf_mean_imputed.predict_proba(X_val_mean_imputed)[:, 1]
        valid_c_index = cindex(y_val, y_val_best)
       
        y_test_imp = rf_mean_imputed.predict_proba(self.X_test)[:, 1]
        test_c_index = cindex(self.y_test, y_test_imp)
       
        return train_c_index, valid_c_index, test_c_index

    
    def iterative_impute(self):
        """
        Function to compute c-index and best hyperparameters for given dataset.
        Create a Linear regression model with the available(non NaN) values, predict the missing values using linear regression
        predictor, train a random forest classifier.

        returns:
        c-index of the dataset of patients calculated after iterative imputation.
        """
        X_train, X_val, y_train, y_val = train_test_split(self.X_dev, self.y_dev, test_size=0.25, random_state=10)
        imputer = IterativeImputer(random_state=0, sample_posterior=False, max_iter=1, min_value=0)
        imputer.fit(X_train)
        X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
        X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
        rf = RandomForestClassifier
        rf_imputed, best_hyperparams_imputed = holdout_grid_search(rf, X_train_imputed, y_train,
                                                           X_val_imputed, y_val,
                                                           hyperparams, {'random_state': 10})

        y_train_best = rf_imputed.predict_proba(X_train_imputed)[:, 1]
        train_c_index = cindex(y_train, y_train_best)
    
        y_val_best = rf_imputed.predict_proba(X_val_imputed)[:, 1]
        valid_c_index = cindex(y_val, y_val_best)
      
        y_test_imp = rf_imputed.predict_proba(self.X_test)[:, 1]
        test_c_index = cindex(self.y_test, y_test_imp)
      
        return train_c_index, valid_c_index, test_c_index