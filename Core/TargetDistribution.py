import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
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
                
