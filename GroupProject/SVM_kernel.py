import sys
import numpy
import pandas
import time
import cvxopt
cvxopt.solvers.options['show_progress'] = False


start_time = time.time() ## starting timer

YX = pandas.read_csv(sys.argv[1]) ##read data
N = len(YX)

Y = YX[YX.columns[13]].to_frame()              ## transform data
X = YX[YX.columns[0:13]]
K = len(X.columns)

F = (X @ X.T + 1)**2
F = pandas.DataFrame(F)


H = (numpy.diagflat(Y.values) @ F @ numpy.diagflat(Y.values)).values  # Hessian

a = numpy.array(cvxopt.solvers.qp(cvxopt.matrix(H, tc='d'),
                                  cvxopt.matrix(-numpy.ones((N,1))),
                                  cvxopt.matrix(-numpy.eye(N)),
                                  cvxopt.matrix(numpy.zeros(N)),
                                  cvxopt.matrix(Y.T.values, tc='d'),
                                  cvxopt.matrix(numpy.zeros(1)))['x'])

i = numpy.argmax(a*Y)

b = Y.T[i] - (Y * a).T @ F[i]

Yhat = numpy.sign(F @ (Y*a) - numpy.ones((N,1)) * b.values)
predictions = numpy.sign(F @ (Yhat*a) - numpy.ones((N,1)) * b.values)
#print(Yhat.to_numpy())

end_time = time.time()

