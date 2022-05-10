import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from .Integration import rk4step
from .Graphix import graphix 
from .Utils import Expect
from abc import ABCMeta, abstractmethod

getcontext().prec = 6
                
class VGP_bank:
    def __init__(self,target,listMean,listCov,beta):
        self.Nfilters=listMean.shape[0]
        self.d=listMean[0].shape[0]
        self.beta=beta
        self.listVGP=[]
        self.target=target
        for i in range(0,self.Nfilters):
            mu0=listMean[i]
            P0=listCov[i]
            filterVGP = VGP_JKO(self.target,mu0,P0,self.beta)
            self.listVGP.append(filterVGP)
        
    def runFilters(self,dt,T):
        for i in range(0,self.Nfilters):
            print("propagate Gaussian ",i)
            self.listVGP[i].propagate(dt,T)
    
        
class VariationalGaussianProcess:
    def __init__(self,target,mean0,cov0):
        self.target=target
        self.mean=mean0
        self.cov=cov0
        #self.noiseMatrix=noiseMatrix
        self.time=0
        self.traj_mean=[]
        self.traj_mean.append(self.mean.reshape(-1,1))
        self.traj_cov=[]
        self.traj_cov.append(self.cov)
    
    @abstractmethod
    # Propagation for a step
    def stepForward(self,dt):
        return

    def propagate(self,dt,T):
        while self.time < T:    
            #print(self.time)
            self.time=self.time+Decimal(dt)
            
            # Kalman propagation step
            self.stepForward(dt)
            
            self.traj_mean.append(self.mean)
            self.traj_cov.append(self.cov)
        return np.asarray(self.traj_mean), np.asarray(self.traj_cov)

    def plot(self,ax,t):
        if t<self.time:
            mean=self.traj_mean[t]
            cov=self.traj_cov[t]
            graphix.plot_ellipsoid2d(ax,mean,cov,'r',zorder=3,linestyle='-',linewidth=2)
    
class VGP_JKO(VariationalGaussianProcess):
    
    def __init__(self,target,mean0,cov0,beta):
        super().__init__(target,mean0,cov0)
        self.R=LA.cholesky(cov0)
        self.invbeta=1/beta
    
    def GaussianDynamicSQRT(X,d,f,invBeta):
        mu=X[0:d].reshape(d,1)
        R=X[d:].reshape(d,d)
        # compute MEAN update
        dmu=Expect.fmeanSQRT(f,mu,R)
        dmu=dmu.reshape(d,1)
        
        # compute SQRT update
        A=Expect.fmeanSQRT(Expect.ExJf,mu,R,mu,f)
        XR=A+A.T+np.identity(d)*invBeta
        #XR=XR*2*invBeta
        invR=LA.inv(R)
        L=Expect.lower(invR.dot(XR).dot(invR.T))
        dR=R.dot(L)
        dR=dR.reshape(-1,1)
        dX=np.concatenate((dmu, dR), axis=0)
        return dX
    
    # Propagation for a step
    def stepForward(self,dt):
        #print("propag ",self.time)
        #print("mean=",self.mean)
        #print("cov=",self.cov)
        d=self.mean.shape[0]
        #D=predictor.diffusionMatrix(self.mean,self.time)
        #Q=predictor.covarianceMatrix()
        #DQD=D.dot(Q).dot(D.T) 
        
        # Runge Kutta integration 
        X0=np.concatenate((self.mean.reshape(-1,1), self.R.reshape(-1,1)), axis=0)
        Xt=rk4step(VGP_JKO.GaussianDynamicSQRT,dt,X0,d,self.target.gradient,self.invbeta)
        self.mean=Xt[0:d]
        self.R=Xt[d:].reshape(d,d)
        self.cov=self.R.dot(self.R.T)

        return self.mean, self.cov

