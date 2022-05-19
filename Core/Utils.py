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

def sigmoid(x):
    #x=np.clip(x,-100,100)
    return 1/(1+np.exp(-x))

def logpdf(theta,X,Y,Z=1):
    theta=theta.reshape(-1,1)
    N,d=X.shape
    # we use the relation logloss=sigmoid(y.theta.u)
    # and log sigmoid(y.theta.u) = - log(1+exp(-y.theta.u)=- logexp(0,-y.theta.u)
    Yb=2*Y.reshape(N,1)-1
    log_pdf_likelihood=np.sum(-np.logaddexp(0, -Yb*X.dot(theta)),axis=0)
    return log_pdf_likelihood-math.log(Z)

# regularized logistic (easier to optimize with Laplace)
def Bayeslogpdf(theta,X,Y,Z,theta0,Cov0):
    (sign, lodetCov0) = LA.slogdet(Cov0)
    invCov0=LA.inv(Cov0)
    theta=theta.reshape(-1,1)
    theta0=theta0.reshape(-1,1)
    e=theta-theta0
    d=theta.shape[0]
    log_pdf_prior=-0.5*(e.T.dot(invCov0).dot(e))-0.5*d*math.log(2*math.pi)+0.5*lodetCov0
    return logpdf(theta,X,Y,Z)+log_pdf_prior.reshape(1,)

def KLnormalVslogpdf(mean,R,logPdf):
     d=mean.shape[0]
     mean=mean.reshape(d,1)
     (sign, logdet) = LA.slogdet(R.dot(R))
     entropy=0.5*logdet+d/2*(1+math.log(2*math.pi))
     ELogp=Expect.fmeanCKF(logPdf,mean,R)
     KL=-ELogp-entropy
     return KL.item()
 
# MC version of KL
def KLnormalVslogpdfMC(mean,R,logPdf,normalSamples):
     d=mean.shape[0]
     mean=mean.reshape(d,1)
     (sign, logdet) = LA.slogdet(R.dot(R))
     entropy=0.5*logdet+d/2*(1+math.log(2*math.pi))
     A=0
     x=mean+R.dot(normalSamples.T)
     cmpt=0
     nbSamplesKL=normalSamples.shape[0]
     for i in range(0,nbSamplesKL):
         xi=x[:,i].reshape(d,1)
         A=A-logPdf(xi)
     KL=A/nbSamplesKL-entropy
     return KL.item()
    
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
    
    def fmeanUKF(f,mu,sqrtP,*args):
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
        d=sqrtP.shape[0]
        fmean=0
        
        for i in range(0,d):
            vi=sqrtP[:,i].reshape(d,1)
            sigmaPointi_Plus=mu+vi*math.sqrt(d)
            sigmaPointi_Moins=mu-vi*math.sqrt(d)
            Wi=1/(2*d)
            fmean=fmean+Wi*f(sigmaPointi_Plus,*args)+Wi*f(sigmaPointi_Moins,*args)
        return fmean
    
    def fmeanMC(f,mu,sqrtP,Nsamples=1,*args):
        P=sqrtP.dot(sqrtP.T)
        mu=mu.reshape(-1,1)
        fmean=0
        Nsamples=Nsamples#100000
        for i in range(0,Nsamples):
            Xi=np.random.multivariate_normal(mu.reshape(-1,),P)
            Xi=Xi.reshape(-1,1)
            fmean=fmean+f(Xi,*args)
        return fmean/Nsamples
    
    def PlotCKFvsMC(f,mu,R,num,text):
        expectMC=[]
        d=mu.shape[0]
        idx=range(1,5000,100)
        for nbMC in idx:
            e=Expect.fmeanMC(f,mu,R,Nsamples=nbMC)
            expectMC.append(e)
        e2=Expect.fmeanCKF(f,mu,R)
        expectCKF=np.ones([len(expectMC),1])*e2
        e3=Expect.fmeanUKF(f,mu,R)
        expectUKF=np.ones([len(expectMC),1])*e3
        fig, ax = plt.subplots(1, 1,figsize=(6,6),num=num)
        ax.plot(idx,expectMC,label="MC")
        ax.plot(idx,expectCKF,label="CKF")
        #ax.plot(idx,expectUKF,label="UKF")
        plt.title("Cubature rules in dimension {} vs MC \n".format(d) + text + "\n CKF use {} sigma points".format(2*d))
        plt.xlabel("number of Monte Carlo samples",fontsize=20)
        plt.ylabel("Expectation value",fontsize=20)
        plt.legend()
        plt.show()
    
    def Exf(x,xmean,f,*args):
        return (x-xmean)*f(x,*args)
    
    def ExJf(x,xmean,Jf,*args):
        return (x-xmean).dot(Jf(x,*args).T)
    
    def Exxf(x,xmean,f,*args):
        return (x-xmean).dot((x-xmean).T)*f(x,*args)
    
    

        