import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from .Integration import rk4step
from .Graphix import graphix 
from .Utils import logpdf
from .SyntheticDataset import LogisticRegDataset, LogisticRegDatasetHD
from abc import ABCMeta, abstractmethod
from scipy.stats import multivariate_normal

getcontext().prec = 6


def sigmoid(x):
    #x=np.clip(x,-100,100)
    return 1/(1+np.exp(-x))

class LangevinTarget():
    def __init__(self,Z=1):
        self.Z=Z
        self.empriricMean=None
        self.empiricalCov=None
        
    def mean(self):
        return
    
    def pdf(self,x):
        return
    
    def logpdf(self,x):
        return
    
    def random(self):
        if self.Gridpdf is None:
            print("no random function available without Grid")
        else:
            nx,ny=self.Gridpdf.shape
            weights=self.Gridpdf.reshape(nx*ny,)
            weights=weights/weights.sum()
            k=np.random.choice(np.arange(0,nx*ny), 1, p=weights)
            i=int(k/nx)+1
            j=k-i*nx
            theta=np.zeros((2,1))
            theta[0]=self.xv[i,j]
            theta[1]=self.yv[i,j]
        return theta
    
    # gradient of log p
    def gradient(self,theta):
        return
    
    def ComputeGrid2D(self,radius,Npoints,center=[]):
        if len(center)==0:
            center=self.mean()
            
        if self.d != 2:
            print("method only valid in dim 2")
            
        self.empriricMean=np.zeros((2,1))
        self.empriricCov=np.zeros((2,2))
        theta1=np.linspace(center[0]-radius,center[0]+radius,Npoints)
        dx=theta1[1]-theta1[0]
        theta2=np.linspace(center[1]-radius,center[1]+radius,Npoints)
        dy=theta2[1]-theta2[0]
        self.Gridpdf=np.zeros((Npoints,Npoints))    
        self.xv,self.yv=np.meshgrid(theta1,theta2)
        Zd=0 #discrete normalization factor
        self.Z=0 # continuous normalization factor (integral)
        for i in np.arange(0,Npoints):
            for j in np.arange(0,Npoints):
                theta=np.zeros((2,1))
                theta[0]=self.xv[i,j]
                theta[1]=self.yv[i,j]
                self.Gridpdf[i,j]=self.pdfUnnormalized(theta)
                Zd=Zd+self.Gridpdf[i,j]
                self.Z=self.Z+self.Gridpdf[i,j]*dx*dy
                self.empriricMean=self.empriricMean+theta*self.pdfUnnormalized(theta)
                self.empriricCov=self.empriricCov+theta.dot(theta.T)*self.pdfUnnormalized(theta)
        self.empriricCov-self.empriricMean.dot(self.empriricMean.T)
        self.empriricMean=self.empriricMean/Zd
        self.empriricCov=self.empriricCov/Zd
        self.Gridpdf=self.Gridpdf/Zd
    
    def plot(self,ax,n_contours=20):
        if not (self.Gridpdf is None):
            CS=ax.contour(self.xv,self.yv,self.Gridpdf,n_contours,zorder=1,\
                          extent=(self.xv[0,0],self.xv[0,-1],self.yv[0,0],self.yv[-1,0]))
   
    def plot3D(self,ax,label=""):
        if not (self.Gridpdf is None):
            ax.plot_surface(self.xv,self.yv,self.Gridpdf,label=label,rstride=1, cstride=1,
                cmap='jet', edgecolor='none',zorder = 0.3,alpha=0.5)
            #ax.plot_wireframe(self.xv,self.yv,self.Gridpdf,label=label,rstride=10, cstride=10)
    
    def plotEllipsoid(self,ax,nbLevels=1,u=0,v=1,label="",col='r'):
        if not (self.empriricMean is None or self.empriricCov is None):
            d=self.empriricMean.shape[0]
            if label != "":
                graphix.plot_ellipsoid2d(ax,self.empriricMean,self.empriricCov,\
                                         col=col,linewidth=1.2,zorder=3,linestyle='-',label=label)
            else:
                graphix.plot_ellipsoid2d(ax,self.empriricMean,self.empriricCov,\
                                         col=col,linewidth=1.2,zorder=3,linestyle='-')
            ax.scatter(self.empriricMean[0],self.empriricMean[1],color=col)
            
    def plotTrueMoments3D(self,ax,xv,yv,label=""):
        gridpdf=np.zeros((xv.shape[0],yv.shape[0]))  
        Z=0
        for i in np.arange(0,xv.shape[0]):
            for j in np.arange(0,yv.shape[0]):
                theta=np.zeros((2,1))
                theta[0]=xv[i,j]
                theta[1]=yv[i,j]
                gridpdf[i,j]=multivariate_normal.pdf(theta.reshape(-1,),self.empriricMean.reshape(-1,),self.empriricCov)
                Z=Z+gridpdf[i,j]
        gridpdf=gridpdf/Z
        ax.plot_wireframe(xv,yv,gridpdf,label=label,rstride=10, cstride=10,zorder = 0.5)
        
        
# A structure to store the parameters of a GMM
class logisticpdf(LangevinTarget):
    def __init__(self,s,N,d,Z=1):
        super().__init__(Z)
        self.N=N
        self.d=d
        self.s=s
        self.Obs=LogisticRegDataset(s,N,d,1,1,scale=1,rotate=True,normalize=True)
    
    def logpdf(self,theta):
        Y,X=self.Obs.datas
        return logpdf(theta,X,Y,self.Z) # for KL plot in 2D use self.Z
        #return logpdf(theta,X,Y,1) # for KL plot in HD use Z=1
            
    def pdf(self,theta):
        return math.exp(self.logpdf(theta))
    
    def pdfUnnormalized(self,theta):
        return math.exp(self.logpdfUnnormalized(theta))
    
    # gradient of log p
    def gradient(self,theta):
        Y,X=self.Obs.datas
        theta=theta.reshape(self.d,1)
        return X.T.dot(Y-sigmoid(X.dot(theta)))

    def plotObs(self,ax,scale=1):
        return self.Obs.plot(ax,scale=scale)
    
    def mean(self):
        return self.empriricMean
    
class logisticpdfHD(logisticpdf):
    def __init__(self,s,N,d,Z):
        super().__init__(s,N,d,Z)
        self.Obs=LogisticRegDatasetHD(s,N,d)
        #self.Obs=LogisticRegObservations(s,N,d,1,1,scale=1,isotropic=True,rotate=False,normalize=True)