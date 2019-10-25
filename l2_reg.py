import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess

class L2_Reg:

    def __init__(self,lr,iter,lmbd):
        self.iter = iter
        self.lr=lr
        self.lmbd = lmbd

    def calc_cost(self,X,Y,theta):
        predictions = []
        for x in X:
            predictions.append(np.dot(x,theta))
        cost = 0.0
        for i in range(len(predictions)):
            cost = cost + (predictions[i]-Y[i])**2
        cost = cost/(2*len(Y))
        return cost

    def calc_gradient(self,X,Y,predictions,theta,index):
        gd = 0.0
        for i in range(len(X)):
            gd = gd + (predictions[i] - Y[i])*X[i][index]
        gd = (gd+self.lmbd*theta[index])
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
                theta[i] = theta[i] - self.lr*self.calc_gradient(X,Y,predictions,theta,i)
            itr_hist.append(it)
            cost_hist.append(self.calc_cost(X,Y,theta))
            if it%5 == 0:
                print("Iteration :",it,"  Cost:",self.calc_cost(X,Y,theta))
        return theta, itr_hist, cost_hist

    def get_coeff(self):
        X,Y,x_mean,y_mean,x_std,y_std = preprocess("3D_spatial_network.csv")
        theta = np.random.rand(3)
        print(theta)
        # print(X)
        ntheta, ith, coh = self.grad_desc(X,Y,theta)
        print(ntheta)
        plt.plot(ith,coh)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Gradient Desc")
        plt.show()

if __name__ == "__main__":
    l2r = L2_Reg(0.000005,50,0.1)
    l2r.get_coeff()
