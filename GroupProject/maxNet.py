import sys
import numpy
import pandas
import time


def logistic( Wx ):
    return numpy.exp( Wx ).div ( numpy.ones(len(Wx)) @ numpy.exp( Wx ) )


YX = pandas.read_csv( sys.argv[1] )           #read data
columns = YX[YX.columns[0]].unique()

Y = pandas.get_dummies( YX[YX.columns[13]] )   #transform data
X = YX[YX.columns[0:13]]
N = len(YX)
X['line'] = numpy.ones((N,1))

I = len(Y.columns)  ## output layer
J = 5               ## hidden layer
K = len(X.columns)  ## input layer
L = 2               ## number of levels

W = [None, pandas.DataFrame(numpy.random.rand(J,K), range(J),  X.columns),
      pandas.DataFrame(numpy.random.rand(I,J), Y.columns, range(J))]

f = {}
df_dWf = {}

fx = {}
dC_dWf = {}

f[0] = lambda x : x
f[1] = lambda x: numpy.maximum( W[1] @ f[0](x), 0)
f[2] = lambda x : logistic( W[2] @ f[1](x) )

for i in range(2):
    print("Epoch {}".format(i))
    for n in range(N):
        for l in range(L + 1):
            fx[l] = f[l](X.iloc[[n]].T)
            if l == 1: df_dWf[l] = numpy.diagflat(numpy.maximum(numpy.sign(W[l] @ fx[l -1]), 0).values )
            if l == 2: df_dWf[l] = ( ( numpy.eye(len(W[l]))
                                       - logistic(W[l] @ fx[l-1]) @ numpy.ones((1, len(W[l]))))
                                      @ numpy.diagflat( logistic( W[l] @ fx[l-1]).values ))
        for l in range(L,0,-1):
            if l==L: dC_dWf[l] = ( logistic( W[l] @ fx[l-1] ) - Y.iloc[[n]].T).T
            else: dC_dWf[l] = dC_dWf[l+1] @ W[l+1] @ df_dWf[l]
            W[l] = W[l] - (1/N) * dC_dWf[l].T @ fx[l-1].T

for n in range(len(YX)):
    print(f[2](X.iloc[[n]].T))
    #print(f[L](X.iloc[[n]].T))