import numpy as np
import pandas as pd
from preprocess import preprocess
import matplotlib.pyplot as plt


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
        for it in range(self.iter):
            for i in range(m):
                ind = np.random.randint(0,m)
                pred = np.dot(X[ind],theta)
                for t in range(3):
                    theta[t] = theta[t] - self.lr*(pred-Y[ind])*X[ind][t]
            itr_hist.append(it)
            cost_hist.append(self.calc_cost(X,Y,theta))
            if it%5 == 0:
                print("Iteration :",it,"  Cost:",self.calc_cost(X,Y,theta))
        return theta, itr_hist, cost_hist

    def get_coeff(self):
        X,Y,x_mean,y_mean,x_std,y_std = preprocess("3D_spatial_network.csv")
        theta = np.random.rand(3)
        print(theta)
        ntheta, ith, coh = self.grad_desc(X,Y,theta)
        print(ntheta)
        plt.plot(ith,coh)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Gradient Desc")
        plt.show()

if __name__ == "__main__":
    sgd = Stoc_Grad_Desc(0.000005,50)
    sgd.get_coeff()
