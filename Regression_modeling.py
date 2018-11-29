import sys
import math
import pandas as pd
import numpy as np
import scipy.stats as ss
import sklearn.model_selection as ms
import sklearn.metrics as sklm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor as rfr
from xgboost import XGBRegressor as xgr
from sklearn.ensemble import AdaBoostRegressor as abr
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet

#######################################################################################################################
def get_regression_model(algo, poly_Order=2, **kwargs):
    # for key in kwargs: print(key) sys.exit()
    print_mod_info = False
    ### Regression models
    ### https://stackoverflow.com/questions/12860841/python-import-in-if

    if   algo == 'XGR': 
        mod = xgr(**kwargs)
    elif algo == 'RFR': 
        mod = rfr(**kwargs)
    elif algo == 'ABR': 
        mod = abr(**kwargs)
    elif algo == 'P1R': 
        mod = LinearRegression(**kwargs)
    elif algo == 'P2R': 
        mod = make_pipeline(PolynomialFeatures(poly_Order), Ridge(**kwargs))
    elif algo == 'ANN': 
        mod = MLPRegressor(**kwargs)
    elif algo == 'ELN':
        mod = ElasticNet(**kwargs) # add parameters later
    elif algo == 'E2R': 
        mod = make_pipeline(PolynomialFeatures(poly_Order), ElasticNet(**kwargs))
    elif algo == 'PLS':
        mod = PLSRegression(**kwargs)
    else: 
        print('Algorithm has not yet been added to the menu.')
        sys.exit()

    return mod

#######################################################################################################################
print("\n#####################################################################################")
# algo = 'P1R'

# 
algo_list = ['P1R', 'P2R', 'RFR', 'ABR', 'XGR', 'ANN', 'ELN', 'E2R', 'PLS']
for algo in algo_list:
    mod = get_regression_model(algo)
    print(mod)
    print('\n')

# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

# Pipeline(memory=None,
#      steps=[('polynomialfeatures', PolynomialFeatures(degree=2, include_bias=True, 
#                                                       interaction_only=False)), 
#             ('ridge', Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#                             normalize=False, random_state=None, solver='auto', tol=0.001))])

# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
#            oob_score=False, random_state=None, verbose=0, warm_start=False)

# AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear',
#          n_estimators=50, random_state=None)

# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#        max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#        silent=True, subsample=1)


# MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(100,), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=200, momentum=0.9,
#        n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
#        random_state=None, shuffle=True, solver='adam', tol=0.0001,
#        validation_fraction=0.1, verbose=False, warm_start=False)


# ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
#       max_iter=1000, normalize=False, positive=False, precompute=False,
#       random_state=None, selection='cyclic', tol=0.0001, warm_start=False)


# Pipeline(memory=None,
#      steps=[('polynomialfeatures', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), ('elasticnet', ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
#       max_iter=1000, normalize=False, positive=False, precompute=False,
#       random_state=None, selection='cyclic', tol=0.0001, warm_start=False))])


# PLSRegression(copy=True, max_iter=500, n_components=2, scale=True, tol=1e-06)