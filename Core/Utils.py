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

# Tools to compute expectations with sigma points
class Expect:
    def lower(A):
        B=np.tril(A)
        B=B-0.5*np.diag(np.diag(A))
        return B
    
    def fmeanSQRT(f,mu,sqrtP,*args):
        n=mu.shape[0]
        mu=mu.reshape(-1,1)
                
        k=3-n
        W0=2*k/(2*(n+k))
        fmean=W0*f(mu,*args)
        
        for i in range(0,n):
            vi=sqrtP[:,i].reshape(n,1)
            sigmaPointi_Plus=mu+vi*math.sqrt(n+k)
            sigmaPointi_Moins=mu-vi*math.sqrt(n+k)
            Wi=1/(2*(n+k))
            fmean=fmean+Wi*f(sigmaPointi_Plus,*args)+Wi*f(sigmaPointi_Moins,*args)
        return fmean
    
    def fmeanCKF(f,mu,sqrtP,*args):
        mu=mu.reshape(-1,1)
        n=sqrtP.shape[0]
        fmean=0
        
        for i in range(0,n):
            vi=sqrtP[:,i].reshape(n,1)
            sigmaPointi_Plus=mu+vi*math.sqrt(n)
            sigmaPointi_Moins=mu-vi*math.sqrt(n)
            Wi=1/(2*n)
            fmean=fmean+Wi*f(sigmaPointi_Plus,*args)+Wi*f(sigmaPointi_Moins,*args)
        return fmean
    
    def fmeanMC(f,mu,sqrtP,Nsamples=1,*args):
        P=sqrtP.dot(sqrtP.T)
        mu=mu.reshape(-1,1)
        fmean=0
        Nsamples=1#100000
        for i in range(0,Nsamples):
            Xi=np.random.multivariate_normal(mu.reshape(-1,),P)
            Xi=Xi.reshape(-1,1)
            fmean=fmean+f(Xi,*args)
        return fmean/Nsamples
    
    def Exf(x,xmean,f,*args):
        return (x-xmean)*f(x,*args)
    
    def ExJf(x,xmean,Jf,*args):
        return (x-xmean).dot(Jf(x,*args).T)
    
    def Exxf(x,xmean,f,*args):
        return (x-xmean).dot((x-xmean).T)*f(x,*args)
    

        