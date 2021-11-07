from linear_utils import lr_model, cindex, add_interactions, make_standard_normal
from random_forest_utils import load_data


class Linear_Model():
    def __init__(self):
        X_train, X_test, y_train, y_test = load_data(10)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        

    def preprocess(self):
        """
        Function used to drop all missing values from the data and normalize the data.

        returns:
        X_train_dropped -- preprocessed training data.
        X_test_dropped -- preprocessed test data.
        y_train_dropped -- training labels.
        y_test_dropped -- test labels.
        """
        X_train_dropped = self.X_train.dropna(axis='rows')
        y_train_dropped = self.y_train.loc[X_train_dropped.index]
        X_test_dropped = self.X_test.dropna(axis='rows')
        y_test_dropped = self.y_test.loc[X_test_dropped.index]
        X_train_dropped, X_test_dropped = make_standard_normal(X_train_dropped, X_test_dropped)
        return X_train_dropped, X_test_dropped, y_train_dropped, y_test_dropped


    def linear(self):
        """
        Function that loads a linear regression model that fits on training data. Calculate the c-index using training labels
        and predicted values.

        returns:
        c-index for train and test sets.
        """
        X_train, X_test, y_train, y_test = self.preprocess()
        model_X = lr_model(X_train, y_train)
        scores = model_X.predict(X_test)[:, 1]
        c_index_X_test = cindex(y_test.values, scores)
        scores2 = model_X.predict(X_train)[:,1]
        c_index_X_train = cindex(y_train.values,scores2)
        return c_index_X_test, c_index_X_train


    def linear_interactions(self):
        """
        Function that loads a linear regression model that fits on training data with interactions of 2 features.
        Calculate the c-index using training labels and predicted values.

        returns:
        c-index for train and test sets.
        """
        X_train, X_test, y_train, y_test = self.preprocess()
        X_train_int = add_interactions(X_train)
        X_test_int = add_interactions(X_test)
        model_X_int = lr_model(X_train_int, y_train)
        scores_X_int = model_X_int.predict_proba(X_test_int)[:, 1]
        c_index_X_int_test = cindex(y_test.values, scores_X_int)
        scores_X_int_2 = model_X_int.predict_proba(X_train_int)[:, 1]
        c_index_X_int_train = cindex(y_train.values, scores_X_int_2)
        return c_index_X_int_test, c_index_X_int_train


