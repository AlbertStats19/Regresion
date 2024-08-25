import numpy as np
import unittest
from utils2 import umbral, model_predictions, load_data_python

import warnings
warnings.filterwarnings("ignore")

Train_Data = False

class prueba(unittest.TestCase):
    def test_mejor_modelo(self):
        yhat_train1, yhat_test1, yhat_train2, yhat_test2 = model_predictions(0.5,parametro=Train_Data)
        X_train, X_test, y_train, y_test = load_data_python()
        if Train_Data==True:
            a = umbral(y_test,yhat_test1)>umbral(y_test,yhat_test2)
        else:
            a = umbral(y_train,yhat_test1)>umbral(y_train,yhat_test2)
        if a==True:
            b = f"El mejor modelo es sin intercepto"
        else:
            b = f"El mejor modelo es con intercepto"

        if (Train_Data==True) & (a==True):
            u1 = umbral(y_train,yhat_train1) 
            u2 = umbral(y_test,yhat_test1)
        elif (Train_Data==True) & (a==False):
            u1 = umbral(y_train,yhat_train2) 
            u2 = umbral(y_test,yhat_test2) 
        elif (Train_Data==False) & (a==False):
            u1 = umbral(y_test,yhat_train2)
            u2 = umbral(y_train,yhat_test2)
        else:
            u1 = umbral(y_test,yhat_train1)
            u2 = umbral(y_train,yhat_test1)

        valores = [u1,u2,-np.abs(u1-u2)]
        comparacion = [80,80,-5]
        for i,j in zip(valores,comparacion):
            self.assertGreaterEqual(i,j)
        return b

pru = prueba()
print(pru.test_mejor_modelo())

if __name__ == "__main__":
    unittest.main()

