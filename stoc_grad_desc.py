import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt


class Stoc_Grad_Desc:

    def __init__(self,lr,iterations):
        self.iter = iterations
        self.lr = lr

    def calc_cost(self,X,Y,theta):
        predictions = []
        for x in X:
            predictions.append(np.dot(x,theta))
        cost = 0.0
        for i in range(len(predictions)):
            cost = cost + (predictions[i]-Y[i])**2
        cost = cost/(2*len(Y))
        return cost

    def grad_desc(self,X,Y,theta):
        cost_hist = []
        m=len(Y)
        itr_hist = []
        prev_cost = 0
        for it in range(self.iter):
            for i in range(m):
                ind = np.random.randint(0,m)
                pred = np.dot(X[ind],theta)
                for t in range(3):
                    theta[t] = theta[t] - self.lr*(pred-Y[ind])*X[ind][t]
            curr_cost = self.calc_cost(X,Y,theta)
            itr_hist.append(it)
            cost_hist.append(curr_cost)
            if it%3 == 0:
                print("Iteration :",it,"  Cost:",curr_cost)
            if abs(curr_cost - prev_cost) < 0.0000001:
                print("Final Cost:",curr_cost)
                break
            prev_cost = curr_cost
        return theta, itr_hist, cost_hist

    def get_coeff(self):
        X,Y,x_mean,y_mean,x_std,y_std = preprocess("3D_spatial_network.csv")
        theta = np.random.rand(3)
        print(theta)
        ntheta, ith, coh = self.grad_desc(X,Y,theta)
        print(ntheta)
        predictions = []
        for x in X:
            predictions.append(np.dot(x,ntheta))
        print("RMSE: ",sqrt(mean_squared_error(Y, predictions)))
        print("R2 Score: ",r2_score(Y,predictions))
        plt.plot(ith,coh)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Gradient Desc")
        plt.show()

if __name__ == "__main__":
    sgd = Stoc_Grad_Desc(0.000001,50)
    sgd.get_coeff()
