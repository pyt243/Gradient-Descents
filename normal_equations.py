import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt


def normal_equations():
        X,Y,x_mean,y_mean,x_std,y_std,X_test,Y_test = preprocess("3D_spatial_network.csv")
        # print(X[:10])
        a = np.linalg.inv(np.transpose(X).dot(X))
        b = np.transpose(X).dot(Y)
        ntheta = a.dot(b)
        print(ntheta)
        predictions = np.dot(X,ntheta)
        print("RMSE: ",sqrt(mean_squared_error(Y, predictions)))
        print("R2 Score: ",r2_score(Y,predictions))
        predictions = np.dot(X_test,ntheta)
        print("RMSE Test: ",sqrt(mean_squared_error(Y_test, predictions)))
        print("R2 Score Test: ",r2_score(Y_test,predictions))
        return ntheta,X_test,Y_test

if __name__ == '__main__':
        ntheta,X_test,Y_test = normal_equations()
