from sklearn.linear_model import Ridge
import pandas as pd
import numpy.linalg as lg
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

def model(lambda_par,parametro):
    X_train, X_test, y_train, y_test = load_data_python()
    modelo = Ridge(alpha = lambda_par,fit_intercept=parametro)
    modelo.fit(X_train, y_train )
    if parametro==True:
        r1 = [float(modelo.intercept_)] + [x for x in modelo.coef_[0]]
    else:
        r1 = modelo.coef_
    return r1

def coefficientes_pedal(lambda_par,parametro):
    X_train, X_test, y_train, y_test = load_data_python()
    if parametro==True:
        covariables = list(X_train.columns)
        X_barra = np.mean(X_train,axis = 0)
        y_barra = np.mean(y_train,axis = 0)
        X_train_c = X_train - X_barra
        y_train_c = y_train - y_barra
        X_train_c["intercepto"] = 1
        X_train_c = X_train_c.get(["intercepto"]+covariables)
        X_train = X_train_c.to_numpy()
        y_train = y_train_c.to_numpy()
    else:
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
    p = X_train.shape[1]
    Id = np.eye(p)
    prod1 = lg.inv(np.matmul(X_train.transpose(),X_train) + lambda_par * Id)
    prod2 = np.matmul(X_train.transpose(),y_train)
    coef = np.matmul(prod1,prod2)

    if parametro==True:
        coef[0] = y_barra - np.dot(X_barra, coef[1:])

    return coef