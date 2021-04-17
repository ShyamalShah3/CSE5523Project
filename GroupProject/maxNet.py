import sys
import numpy
import pandas
import tensorflow as tf


YX = pandas.read_csv( './Data/heart-b.csv' ) ## read in data
Y = pandas.get_dummies( YX[YX.columns[13]] ) ## transform data
X = YX[YX.columns[:13]]
             ## transform data
# X = YX[YX.columns[0:13]]
N = len(YX)
X['line'] = numpy.ones((N,1))

X0 = tf.constant( X, dtype=tf.float32 ) ## tensorflow format
Y0 = tf.constant( Y, dtype=tf.float32 )

unit = 60
## trainable matrices
W_H = tf.Variable( tf.random.uniform( [unit,unit+len(X.columns)], dtype=tf.float32 ), trainable=True )
W_Y = tf.Variable( tf.random.uniform( [len(Y.columns),unit], dtype=tf.float32 ), trainable=True )
def estimate(): ## calculate estimate
    Yhat = [] ## store output in list
    h = tf.Variable( numpy.zeros((unit,1)), dtype=tf.float32 ) ## initial hidden state
    for t in range(len(YX)): ## iterate over input
        xT = tf.reshape( X0[t], (len(X.columns),1) ) ## update hidden state
        h = tf.nn.softmax(W_H @ tf.concat( [h,xT], axis=0 ), axis=0 )
        Yhat.append( tf.reshape( tf.nn.softmax(W_Y @ h, axis=0), (len(Y.columns),) ) )
    return tf.stack(Yhat) ## output as matrix

def cost(): ## cross entropy cost
    return - 1/N * tf.reduce_sum( tf.math.log( tf.reduce_sum( Y0 * estimate(), axis=1 ) ) )

# opt = tf.keras.optimizers.SGD( 0.01 ) ## specify SGD optimizer
opt = tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False,
    name='Adam'
)

for epoch in range( 150 ): ## for each epoch
    print("Epoch " + str(epoch))
    opt.minimize( cost, var_list=[W_H,W_Y] ) ## do fwd and backprop

print( W_H ) ## output models
print( W_Y )
est = estimate()
c = cost()
# print( est) ## output estimate
# print( c ) ## output final cost

est_round = numpy.round(est)
print( est_round) ## output estimate
print( c ) ## output final cost
# def yVal():
#     y_val = []
#     for i in est_round:
#         if est_round[0].any() > est_round[1].any(): 
#             y_val.append(0)
#         else: 
#             y_val.append(1)
#     return y_val

# y_val = yVal()
# Y = YX[YX.columns[13]].to_frame() 
# count = 0
# for i in range(N):
#     if y_val[i] == Y[i]: count = 1+count
        