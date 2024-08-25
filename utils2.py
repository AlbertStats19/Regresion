from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def load_data_python():
    dataset = pd.read_csv("base.csv",header=0,sep=";",decimal=",")
    prueba = pd.read_csv("prueba.csv",header=0,sep=";",decimal=",")
    covariables = [x for x in dataset.columns if x not in ["y"]]
    X_train, y_train = dataset.get(covariables), dataset.get(["y"])
    X_test, y_test = prueba.get(covariables), prueba.get(["y"])
    return X_train, X_test, y_train, y_test

def umbral(ytrue,yhat,u = 0.07):
    return 100 * np.mean(np.abs((ytrue-yhat)/ytrue)<=u)

def model_predictions(lambda_par,parametro):
    X_train, X_test, y_train, y_test = load_data_python()
    if parametro==True:
        modelo = Ridge(alpha = lambda_par,fit_intercept=False)
        modelo.fit(X_train,y_train)
        modelo2 = Ridge(alpha = lambda_par,fit_intercept=True)
        modelo2.fit(X_train,y_train)
        yhat_train1 = modelo.predict(X_train)
        yhat_test1 = modelo.predict(X_test)
        yhat_train2 = modelo2.predict(X_train)
        yhat_test2 = modelo2.predict(X_test)
    else:
        modelo = Ridge(alpha = lambda_par,fit_intercept=False)
        modelo.fit(X_test,y_test)
        modelo2 = Ridge(alpha = lambda_par,fit_intercept=True)
        modelo2.fit(X_test,y_test)
        yhat_train1 = modelo.predict(X_test)
        yhat_test1 = modelo.predict(X_train)
        yhat_train2 = modelo2.predict(X_test)
        yhat_test2 = modelo2.predict(X_train)
    return yhat_train1, yhat_test1, yhat_train2, yhat_test2

