# -*- coding: utf-8 -*-

#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
'''
Description: 
    This module provides a sample matrix factorization example
@author: jason
'''


try:
    import numpy
except:
    print "This implementation requires the numpy module."
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    error = 0.0
    Q = Q.T
    for step in xrange(steps):
        
        # update
        for i in xrange(len(R)): # row
            for j in xrange(len(R[i])): # column
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        
        # check convergence
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
                        
        print ("step: %d, error: %.4f" % (step, e))
        print "P="
        print P
        print "Q="
        print Q
        print "------------------"
        
        if e < 0.001:
            error = e
            break
    return P, Q.T, error

def matrix_factorization_ex(R, P, Q, K, steps=1000, alpha=0.002, beta=0.02):
    Q = Q.T
#     Indi = numpy.copy(R)
#     Indi[Indi<>0] = 1
    Indi = numpy.where(R<>0, 1, 0)
    for step in xrange(steps):
        Pred = P.dot(Q)
        _Pred = numpy.multiply(Indi, Pred)
        E = R -  _Pred
        P_tmp = numpy.copy(P)
        Q_tmp = numpy.copy(Q)
        P = P_tmp + alpha*(E.dot(Q.T) - beta*P_tmp)
        Q = Q_tmp + alpha*(P.T.dot(E) - beta*Q_tmp)
        rmse = numpy.sqrt(E.ravel().dot(E.flat) / len(Indi[Indi.nonzero()]))
        print 'step:%s'%step
        print "RMSE:", rmse
        
        if rmse < 0.001:
            break
        
    return P, Q.T

###############################################################################

if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

#     nP, nQ , error = matrix_factorization(R, P, Q, K)
    nP, nQ  = matrix_factorization_ex(R, P, Q, K)
    
    print "P="
    print nP
    print "Q="
    print nQ
