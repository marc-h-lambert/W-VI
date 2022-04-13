import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from .Integration import rk4step
from .Graphix import graphix 
from abc import ABCMeta, abstractmethod
from scipy.stats import multivariate_normal

getcontext().prec = 6

class Distribution:
    def gradient(self,x):
        return
    
class GaussianMixture(Distribution):
    def __init__(self,listw,listMean,listCov):
        self.K=listMean.shape[0]
        self.d=listMean[0].shape[0]
        self.listw=listw
        self.listMean=listMean
        self.listCov=listCov
        
    def pdf(self,x):        
        y=0
        for i in range(0,self.K):
            mean=self.listMean[i]
            cov=self.listCov[i]
            w=self.listw[i].reshape(-1,)
            n=multivariate_normal.pdf(x.reshape(-1,),mean.reshape(-1,),cov)
            y=y+w*n
        return y
    
    def gradient(self,x):
        y=np.zeros([self.d,1])
        for i in range(0,self.K):
            mean=np.array(self.listMean[i]).reshape([-1,1])
            cov=np.array(self.listCov[i])
            w=self.listw[i]
            e=(x-mean).reshape(-1,1)
            n=multivariate_normal.pdf(x.reshape(-1,),mean.reshape([-1,]),cov)
            y=y-w*LA.inv(cov).dot(e)*n
        y=y/self.pdf(x)
        #print("grad=",y)
        return y.reshape([-1,1])
        
    def ComputeGrid2D(self,center,radius,Npoints):
        if self.d != 2:
            print("method only valid in dim 2")
        theta1=np.linspace(center[0]-radius,center[0]+radius,Npoints)
        theta2=np.linspace(center[1]-radius,center[1]+radius,Npoints)
        self.Gridpdf=np.zeros((Npoints,Npoints))    
        self.xv,self.yv=np.meshgrid(theta1,theta2)
        for i in np.arange(0,Npoints):
            for j in np.arange(0,Npoints):
                theta=np.zeros((2,1))
                theta[0]=self.xv[i,j]
                theta[1]=self.yv[i,j]
                self.Gridpdf[i,j]=self.pdf(theta)
                
    def plot(self,ax,n_contours=20):
        if not (self.Gridpdf is None):
            CS=ax.contour(self.xv,self.yv,self.Gridpdf,n_contours,zorder=1,\
                          extent=(self.xv[0,0],self.xv[0,-1],self.yv[0,0],self.yv[-1,0]))
                
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
    
    def fmeanMC(f,mu,sqrtP,*args):
        P=sqrtP.dot(sqrtP.T)
        mu=mu.reshape(-1,1)
        fmean=0
        Nsamples=100000
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

             