from sklearn.linear_model import Ridge
import pandas as pd
import numpy.linalg as lg
import numpy as np
import warnings
import unittest
warnings.filterwarnings("ignore")
from utils1 import load_data_python, model, coefficientes_pedal

parametro = False

class prueba(unittest.TestCase):
    def test_comparar(self):
        cf_funcion = model(lambda_par = 0.5, parametro = parametro)
        if parametro==False:
            cf_funcion = [x for x in cf_funcion[0]]

        cf_pedal = coefficientes_pedal(lambda_par = 0.5,parametro=parametro)
        cf_pedal = cf_pedal.transpose()
        cf_pedal = [x for x in cf_pedal[0]]

        for i,j in zip(cf_funcion,cf_pedal):
            i = np.round(i,2)
            j = np.round(j,2)
            self.assertAlmostEqual(i,j)

if __name__ == "__main__":
    unittest.main()