import numpy as np
import numpy.random
import numpy.linalg as LA
from .Graphix import graphix 
import math
from math import log, exp
from scipy import optimize
from Core.Utils import logpdf,Bayeslogpdf, sigmoid
from scipy.stats import multivariate_normal

def negBaylogpdf(theta,X,Y,Z,theta0,Cov0):
    return -Bayeslogpdf(theta,X,Y,Z,theta0,Cov0)

def neglogpdf(theta,X,Y,Z):
    return -logpdf(theta,X,Y,Z)

# Batch Laplace version of Bayesian Logistic Regression (with a Gaussian model)
class LaplaceLogReg():
    
    def __init__(self,x0,Cov0=None):
        self.x0=x0
        self.Cov0=Cov0
        
    def fit(self,X,Y):   
        N,d=X.shape
        if self.Cov0 is None: 
            sol=optimize.minimize(neglogpdf, self.x0,args=(X,Y,1),method='L-BFGS-B')
        else: # a version regularized with an initial covariance is less prone to diverge
            sol=optimize.minimize(negBaylogpdf, self.x0,args=(X,Y,1,self.x0,self.Cov0),method='L-BFGS-B')
        self.mode=sol.x
        
        # the Hessian 
        L=sigmoid(X.dot(self.mode))
        K=(L*(1-L)).reshape(N,1,1)
        # Tensor version
        #A=X[...,None]*X[:,None]
        #H=np.sum(K*A,axis=0)+LA.inv(self._Cov0)
        # Memory free version
        if self.Cov0 is None: 
            H=np.zeros([d,d])
        else:
            H=LA.inv(self.Cov0)
        #H=np.zeros([d,d])
        for i in range(0,N):
            xt=X[i,:].reshape(d,1)
            H=H+K[i]*xt.dot(xt.T)
        self.Cov=LA.inv(H)
        
        return self
    
    def plotEllipsoid(self,ax,nbLevels=1,u=0,v=1,label="",col='r'):
        d=self.mode.shape[0]
        meanproj,Covproj=graphix.projEllipsoid(self.mode,self.Cov.reshape(d,d),u,v)
        if label != "":
            graphix.plot_ellipsoid2d(ax,meanproj,Covproj,col=col,linewidth=1.2,zorder=3,linestyle='-',label=label)
        else:
            graphix.plot_ellipsoid2d(ax,meanproj,Covproj,col=col,linewidth=1.2,zorder=3,linestyle='-')
        ax.scatter(self.mode[0],self.mode[1],color=col)
        
    def KLseed(self,target,weightSamples,normalSamples,test=True):  
        y=0
        nbMC=weightSamples.shape[0]
        
        for i in range(0,nbMC):
            k=weightSamples[i]
            mu=self.mode.reshape(-1,1)
            R=LA.cholesky(self.Cov)
            u=normalSamples[i].reshape(-1,1)
            x=mu+R.dot(u)
            if test:
                print("R=",R)
                print("x=",x)
            x=x.reshape(-1,)
            n=multivariate_normal.pdf(x.reshape(-1,),self.mode.reshape(-1,),self.Cov)
            print(n)
            y=y+math.log(n)-math.log(target.pdf(x))
            if test:
                print("y=",y)
        return y/nbMC
    
    def plot3D(self,ax,xv,yv,label="",col='b'):
        gridpdf=np.zeros((xv.shape[0],yv.shape[0]))  
        Z=0
        for i in np.arange(0,xv.shape[0]):
            for j in np.arange(0,yv.shape[0]):
                theta=np.zeros((2,1))
                theta[0]=xv[i,j]
                theta[1]=yv[i,j]
                gridpdf[i,j]=multivariate_normal.pdf(theta.reshape(-1,),self.mode.reshape(-1,),self.Cov)
                Z=Z+gridpdf[i,j]
        gridpdf=gridpdf/Z
        ax.plot_wireframe(xv,yv,gridpdf,label=label,rstride=10, cstride=10,zorder = 0.5,color=col)
        
