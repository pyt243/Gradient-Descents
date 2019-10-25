import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

class Grad_Desc:

    def __init__(self,lr,iterations):
        self.lr = lr
        self.iter = iterations

    def calc_cost(self,X,Y,theta):
        predictions = []
        for x in X:
            predictions.append(np.dot(x,theta))
        cost = 0.0
        for i in range(len(predictions)):
            cost = cost + (predictions[i]-Y[i])**2
        cost = cost/(2*len(Y))
        return cost

    def calc_gradient(self,X,Y,predictions,index):
        gd = 0.0
        for i in range(len(X)):
            gd = gd + (predictions[i] - Y[i])*X[i][index]
        gd = gd#/len(Y)
        return gd

    def grad_desc(self,X,Y,theta):
        cost_hist = []
        m=len(Y)
        itr_hist = []
        for it in range(self.iter):
            predictions = []
            for x in X:
                predictions.append(np.dot(x,theta))
            for i in range(3):
                theta[i] = theta[i] - self.lr*self.calc_gradient(X,Y,predictions,i)
            if it%1 == 0:
                itr_hist.append(it)
                cost_hist.append(self.calc_cost(X,Y,theta))
            if it%3 == 0:
                print("Iteration :",it,"  Cost:",self.calc_cost(X,Y,theta))
        return theta, itr_hist, cost_hist


    def get_coeff(self):
        X,Y,x_mean,y_mean,x_std,y_std = preprocess("3D_spatial_network.csv")
        theta = np.random.rand(3)
        print(theta)
        # print(X)
        ntheta, ith, coh = self.grad_desc(X,Y,theta)
        predictions = []
        for x in X:
            predictions.append(np.dot(x,ntheta))
        print(ntheta)
        print("RMSE: ",sqrt(mean_squared_error(Y, predictions)))
        print("R2 Score: ",r2_score(Y,predictions))
        plt.plot(ith,coh)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Gradient Desc")
        plt.show()


if __name__ == "__main__":
    gd = Grad_Desc(0.000005,45)
    gd.get_coeff()
