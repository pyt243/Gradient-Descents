import numpy as np
import pandas as pd

def preprocess(path):
    np.random.seed(42)
    data = pd.read_csv(path)
    Y = data['3'].values
    y_mean = np.mean(Y)
    y_std = np.std(Y)
    Y = Y - y_mean
    Y = Y/y_std
    data = data.drop(['0','3'],axis=1)
    x_mean=[]
    x_std=[]
    for i in range(1,3):
        temp = data[str(i)].values
        data.drop([str(i)],axis=1);
        xm = np.mean(temp)
        xs = np.std(temp)
        temp = temp - xm
        temp = temp/xs
        x_mean.append(xm)
        x_std.append(xs)
        data[str(i)]=temp
    ones = np.ones(len(Y),dtype=float)
    data['0'] = ones
    data['3'] = Y
    sdf = data.values
    np.random.shuffle(sdf)
    data = pd.DataFrame(sdf,columns=['0','1','2','3'])
    # print(data.values[:10])
    Y = data['3'].values
    data = data.drop(['3'],axis=1)
    X = data.values
    X_test = X[:][300000:]
    Y_test = Y[300000:]
    X = X[:][:300000]
    Y = Y[:300000]
    X = np.array(X,dtype=np.float64)
    return X,Y,x_mean,y_mean,x_std,y_std,X_test,Y_test

# preprocess("3D_spatial_network.csv")
