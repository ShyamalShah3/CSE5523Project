import sys
import numpy as np
import pandas
import time


def write_csv(predictions):
    df = pandas.DataFrame(data=predictions,columns=['Regression Predictions'])
    df.to_csv("regressionPredictions.csv", index=False)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost


def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        params = params - ((learning_rate/m) * (X.T @ (sigmoid(X @ params) - y)))

    return params


def predict(X, params):
    return np.round(sigmoid(X @ params))


def accuracy(target, predicted):
    counter = 0
    for i in range(target.shape[0]):
        if target[i,0] == predicted[i,0]:
            counter += 1
    return counter / target.shape[0]


start_time = time.time()
YX = pandas.read_csv(sys.argv[1]) ##read data
N = len(YX)

Y = YX.to_numpy()
Y = Y[:,13].reshape((303,1))
print(Y.shape)
X = YX[YX.columns[0:13]]
X['line'] = np.ones(N)
K = len(X.columns)

weights = np.random.rand(K, 1)
iterations = 1000
learningrate = .01

weights_final = gradient_descent(X.to_numpy(), Y, weights, learningrate, iterations)

predict_target_train = predict(X.to_numpy(), weights_final)
end_time = time.time()
print("Total Time for Regression: {} seconds".format(round((end_time - start_time), 3)))
print(accuracy(Y, predict_target_train))
predictions = predict_target_train.flatten().tolist()
write_csv(predictions)

