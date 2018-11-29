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


def get_regression_model(algo, settings, print_mod_info=False):
    ### Regression models
    ### https://stackoverflow.com/questions/12860841/python-import-in-if
    if   algo == 'XGR': 
        mod = xgr(n_estimators = settings[0], max_depth=settings[1])
        if print_mod_info: print('XGBoost:', mod)
    elif algo == 'RFR': 
        mod = rfr(n_estimators = settings[0])
        if print_mod_info: print('Random Forest:', mod)
    elif algo == 'ABR': 
        mod = abr(n_estimators = settings[0])
        if print_mod_info: print('AdaBoost:', mod)
    elif algo == 'P1R': 
        mod = LinearRegression()
        if print_mod_info: print('Linear:', mod)
    elif algo == 'P2R': 
        mod = make_pipeline(PolynomialFeatures(settings[0]), Ridge())
        if print_mod_info: print('Poly 2:', mod)
    elif algo == 'ANN': 
        mod = MLPRegressor(solver='lbfgs',
                           hidden_layer_sizes=(settings[0],settings[1]), # (137,73), 
                           tol=settings[2])
        if print_mod_info: print('Neural Net Regression:', mod)
    elif algo == 'ELN':
        mod = ElasticNet(alpha=settings[0],l1_ratio=settings[1]) # add parameters later
        if print_mod_info: print('Elastic Net Regression:', mod)
    elif algo == 'E2R': 
        mod = make_pipeline(PolynomialFeatures(settings[0]), ElasticNet(alpha=settings[1],l1_ratio=settings[2]))
        if print_mod_info: print('Poly 2:', mod)
    elif algo == 'PLS':
        mod = PLSRegression(n_components=settings[0])
        if print_mod_info: print('Partial Least Squares Regression:', mod)
    else: 
        print('Algorithm not setup yet.')
        sys.exit()

    return mod

