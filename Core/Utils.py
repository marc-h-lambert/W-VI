###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Mathematical functions                                                          #
###################################################################################

import numpy.linalg as LA 
import numpy as np
import math
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt


def KLDivergence(P0,P1):
    d=P0.shape[0]
    (sign, logdetP0) = LA.slogdet(P0)
    if sign <0:
        print("logdet <0 for P0=",P0)
    (sign, logdetP1) = LA.slogdet(P1)
    if sign <0:
        print("logdet <0 for P1=",P1)
    return 0.5*(logdetP1-logdetP0+np.trace(LA.inv(P1).dot(P0))-d)

#if U N x d, p N x 1 compute Sum pi Ui Ui^T of size d x d
def empiricalCov(U,p):
    M=U[...,None]*U[:,None]
    p=p.reshape(-1,1,1)
    return np.sum(p*M,axis=0)
    

        