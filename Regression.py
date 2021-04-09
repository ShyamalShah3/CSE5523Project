import sys
import numpy as np
import pandas
import math
from csv import writer
import time
from sklearn.model_selection import train_test_split

def sigmoid(x):
    #print(x.shape)
    return 1 / (1 + np.exp(-x))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost

def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))
    for i in range(iterations):
        '''print(X.shape)
        print(params.shape)
        print(y.shape)
        print("---------------")'''
        params = params - ((learning_rate/m) * (X.T @ (sigmoid(X @ params) - y)))
        #cost_history[i] = compute_cost(X, y, params)

    return params

def predict(X, params):
    return np.round(sigmoid(X @ params))

def accuracy(target, predicted):
    counter = 0
    for i in range(target.shape[0]):
        if target[i,0] == predicted[i,0]:
            counter += 1
    return counter / target.shape[0]




YX = pandas.read_csv(sys.argv[1]) ##read data
#YX = YX.astype(dtype="float")
#print(YX)
#columns = YX[YX.columns[-1]].unique()
N = len(YX)

#Y = pandas.get_dummies(YX[YX.columns[13]]) ##one-hot y's
Y = YX.to_numpy()
Y = Y[:,13].reshape((303,1))
print(Y.shape)
X = YX[YX.columns[0:13]]
X['line'] = np.ones(N)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
N = len(X_train)
J = 1
K = len(X_train.columns)

weights = np.random.rand(K, 1)
iterations = 1000
learningrate = .01

weights_final = gradient_descent(X.to_numpy(), Y, weights, learningrate, iterations)

predict_target_train = predict(X.to_numpy(), weights_final)
print(accuracy(Y, predict_target_train))

'''predict_target_test = predict(X_test.to_numpy(), weights_final)
print(predict_target_test.shape)
print(Y_test.shape)
print(accuracy(Y_test, predict_target_test))'''




"""BM = 0.9
BV = 0.999
alpha = 1.0
epsilon = math.pow(10,(-8))
M_subI = 0
V_subI = 0

W = pandas.DataFrame( np.random.rand(J,K), Y_train.columns, X_train.columns ) ## random weights
starttime = time.time()
for i in range(1000):

    derivative = np.exp(W @ X_train.T)
    derivative = np.ones(J).T @ derivative
    derivative = np.diag(derivative)
    derivative = np.linalg.inv(derivative)
    derivative2 = W @ X_train.T
    derivative2 = np.exp(derivative2)
    derivative2 = Y_train.T - derivative2
    derivative = derivative2 @ derivative
    #derivative = (1/N)*derivative
    print(derivative.shape)
    print(X_train.shape)
    egg = derivative @ X_train
    derivative = -(1/N)*egg

    #derivative = -((1/N) * ((Y_train.T - np.exp(W @ X_train.T)) @ (np.linalg.inv(np.diag(np.ones(J).T @ np.exp(W @ X_train.T))))) @ X_train)
    M_subI = (BM * M_subI) + ((1 - BM) * derivative)
    V_subI = (BV * V_subI) + ((1 - BV) * (np.square(derivative)))
    M_hat = M_subI / (1 - (math.pow(BM, i+1)))
    V_hat = V_subI / (1 - (math.pow(BV, i+1)))
    W = W - ((alpha / (np.sqrt(V_hat) + epsilon)) * M_hat)
training_time = round(time.time() - starttime,4)
print("It took {} seconds for the Adam optimizer to be trained".format(training_time)) ##print time

N2 = len(X_test)
J2 = len(Y_test.columns)
K2 = len(X_test.columns)
predictions = ( np.exp(W @ X_test.T) @ np.linalg.inv( np.diag( np.ones(J2).T @ np.exp(W @ X_test.T))) )

write_csv(predictions)
"""