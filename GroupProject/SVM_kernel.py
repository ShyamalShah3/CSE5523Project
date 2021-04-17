import sys
import numpy
import pandas
import time
import cvxopt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler


def sklearnModel():
    # Initialize SVM classifier
    clf = svm.SVC(kernel='linear')
    # Fit data
    clf = clf.fit(X, Y)
    predictions = clf.predict(X)
    temp = Y.to_numpy()
    count = 0
    print(predictions)
    for i in range(N):
        if temp[i] == predictions[i]: count = 1+count
    
    print(str(count/len(Y)))
    
    
    
cvxopt.solvers.options['show_progress'] = False


start_time = time.time() ## starting timer

YX = pandas.read_csv('./Data/heart-b.csv') ##read data
N = len(YX)

Y = YX[YX.columns[13]].to_frame()              ## transform data
Y['target'] = Y['target'].replace([0],-1)
X = YX[YX.columns[0:13]]
K = len(X.columns)

scaler = StandardScaler() 
data_scaled = scaler.fit_transform(X)
X = pandas.DataFrame(data_scaled)

F = (X @ X.T + 1)**3
# F = pandas.DataFrame(F)
# F = X @ X.T

H = (numpy.diagflat(Y.values) @ F @ numpy.diagflat(Y.values)).values  # Hessian

a = numpy.array(cvxopt.solvers.qp(cvxopt.matrix(H, tc='d'),
                                  cvxopt.matrix(-numpy.ones((N,1))),
                                  cvxopt.matrix(-numpy.eye(N)),
                                  cvxopt.matrix(numpy.zeros(N)),
                                  cvxopt.matrix(Y.T.values, tc='d'),
                                  cvxopt.matrix(numpy.zeros(1) ))['x'] )

i = numpy.argmax(a * Y)

b = Y.T[i] - (a * Y).T @ F[i]

Yhat = numpy.sign(F @ (a * Y) - numpy.ones((N,1)) * b.values)
predictions = numpy.sign(F @ (Yhat*a) - numpy.ones((N,1)) * b.values)
#print(Yhat.to_numpy())

# sklearnModel()
print(predictions)
count = 0
for i in range(len(predictions)):
    if predictions['target'][i] == Y['target'][i]: count = 1 + count

print(str(count/N))
    

end_time = time.time()


    
