from linear_model import Linear_Model
from random_forest_model import Random_Forest_Model

LM = Linear_Model()
c_index_test, c_index_train = LM.linear()
c_index_int_test, c_index_int_train = LM.linear_interactions()


RFM = Random_Forest_Model()
c_index_train_dropna, c_index_valid_dropna, c_index_test_dropna = RFM.drop_missing_values()
c_ndex_train_mean, c_index_valid_mean, c_index_test_mean = RFM.mean_impute()
c_index_train_iter, c_index_valid_iter, c_index_test_iter = RFM.iterative_impute()



