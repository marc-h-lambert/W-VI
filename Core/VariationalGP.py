import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from .Integration import rk4step
from .Graphix import graphix 
from .Utils import Expect
from abc import ABCMeta, abstractmethod
from scipy.stats import multivariate_normal

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
            self.time=self.time+Decimal(dt)
            
            # Kalman propagation step
            self.stepForward(dt)
            
            self.traj_mean.append(self.mean)
            self.traj_cov.append(self.cov)
        return np.asarray(self.traj_mean), np.asarray(self.traj_cov)

    def plot(self,ax,t):
        mean=self.traj_mean[t]
        cov=self.traj_cov[t]
        graphix.plot_ellipsoid2d(ax,mean,cov,'r',zorder=3,linestyle='-',linewidth=2)
            
    def plotOptimal(self,ax,label=None):
        if label is None:
            graphix.plot_ellipsoid2d(ax,self.mean,self.cov,'r',zorder=3,linestyle='-',linewidth=2)
        else:
            graphix.plot_ellipsoid2d(ax,self.mean,self.cov,'r',zorder=3,linestyle='-',\
                                     linewidth=2,label=label)
                
    def plot3D(self,ax,xv,yv,label="",col='r'):
        gridpdf=np.zeros((xv.shape[0],yv.shape[0]))  
        Z=0
        for i in np.arange(0,xv.shape[0]):
            for j in np.arange(0,yv.shape[0]):
                theta=np.zeros((2,1))
                theta[0]=xv[i,j]
                theta[1]=yv[i,j]
                gridpdf[i,j]=multivariate_normal.pdf(theta.reshape(-1,),\
                                                     self.mean.reshape(-1,),self.cov)
                Z=Z+gridpdf[i,j]
        gridpdf=gridpdf/Z
        ax.plot_wireframe(xv,yv,gridpdf,label=label,rstride=10, cstride=10,zorder = 0.5,color=col)
    
    def KL(self,t):
        mean=self.traj_mean[t]
        cov=self.traj_cov[t]
        d=mean.shape[0]
        (sign, logdet) = LA.slogdet(cov)
        entropy=0.5*logdet+d/2*(1+math.log(2*math.pi))
        ELogp=Expect.fmeanCKF(self.target.logpdf,mean.reshape(d,1),LA.cholesky(cov))
        KL=-ELogp-entropy
        return KL.item()
    
class VGP_JKO(VariationalGaussianProcess):
    
    def __init__(self,target,mean0,cov0,invbeta):
        super().__init__(target,mean0,cov0)
        self.R=LA.cholesky(cov0)
        self.invbeta=invbeta
    
    def GaussianDynamicSQRT(X,d,f,invBeta):
        mu=X[0:d].reshape(d,1)
        R=X[d:].reshape(d,d)
        # compute MEAN update
        dmu=Expect.fmeanCKF(f,mu,R)
        dmu=dmu.reshape(d,1)
        # compute SQRT update
        A=Expect.fmeanCKF(Expect.ExJf,mu,R,mu,f)
        XR=A+A.T+2*np.identity(d)*invBeta
        #XR=XR*2*invBeta
        invR=LA.inv(R)
        L=Expect.lower(invR.dot(XR).dot(invR.T))
        dR=R.dot(L)
        dR=dR.reshape(-1,1)
        dX=np.concatenate((dmu, dR), axis=0)
        return dX
    
    # Propagation for a step
    def stepForward(self,dt):
        print("propag ",self.time)
        d=self.mean.shape[0]
        
        # Runge Kutta integration 
        X0=np.concatenate((self.mean.reshape(-1,1), self.R.reshape(-1,1)), axis=0)
        Xt=rk4step(VGP_JKO.GaussianDynamicSQRT,dt,X0,d,self.target.gradient,self.invbeta)
        self.mean=Xt[0:d]
        self.R=Xt[d:].reshape(d,d)
        self.cov=self.R.dot(self.R.T)

        return self.mean, self.cov

