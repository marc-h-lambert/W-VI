import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from .Integration import rk4step
from .Graphix import graphix 
from .Utils import Expect
from .LangevinTarget import LangevinTarget
from abc import ABCMeta, abstractmethod
from scipy.stats import multivariate_normal

getcontext().prec = 6

# A structure to store the parameters of a GMM (which can be used also as a target)
class GMM(LangevinTarget):
    def __init__(self,listw,listMean,listSqrt):
        listMean=np.asarray(listMean)
        self.K=listMean.shape[0]
        d=listMean[0].shape[0]
        self.d=d
        self.weights=np.zeros((self.K))
        self.means=np.zeros((self.K,d))
        self.rootcovs=np.zeros((self.K,d,d))
        for k in range(0,self.K):
            self.weights[k]=listw[k]
            self.means[k,:]=listMean[k]
            self.rootcovs[k,:,:]=listSqrt[k]
    
    def mean(self):
        return self.means.mean(axis=0)
    
    def cov2sqrt(listCov):
        listSqrt=[]
        for P in listCov:
            listSqrt.append(LA.cholesky(P))
        return np.array(listSqrt)
    
    # put all the parameters of the mixture in a single vector X
    def GMM2Vector(self):
        X=np.empty((0,1))
        for k in range(0,self.K):
            wk=self.weights[k].reshape(-1,1)
            meank=self.means[k,:].reshape(-1,1)
            Rk=self.rootcovs[k,:,:].reshape(-1,1)
            X=np.concatenate((X,np.log(wk),meank,Rk), axis=0)
        return X
        
     # (static class) retrieve all the parameters of the mixture from a single vector X
    def Vector2GMM(X,d,K):
        weights=np.zeros((K))
        means=np.zeros((K,d))
        rootcovs=np.zeros((K,d,d))
        
        SumWk=0
        idx=0
        for k in range(0,K):
            SumWk=SumWk+math.exp(X[idx])
            idx=idx+1+d+d*d
        idx=0
        for k in range(0,K):
            wk=math.exp(X[idx])/SumWk
            meank=X[idx+1:idx+1+d].reshape(d,)
            Rk=X[idx+1+d:idx+1+d+d*d].reshape(d,d)
            idx=idx+1+d+d*d
            weights[k]=wk
            means[k,:]=meank
            rootcovs[k,:,:]=Rk
        return GMM(weights,means,rootcovs)
    
    def pdf(self,x):        
        y=0
        for k in range(0,self.K):
            w=self.weights[k]
            mu=self.means[k,:]
            R=self.rootcovs[k,:,:]
            P=R.dot(R.T)
            n=multivariate_normal.pdf(x.reshape(-1,),mu.reshape(-1,),P)
            y=y+w*n
        return y
    
    def logpdf(self,x):
        return math.log(self.pdf(x))
    
    def random(self):        
        k=np.random.choice(np.arange(0,self.K), 1, p=self.weights.reshape(-1,))
        k=int(k)
        mu=self.means[k,:]
        R=self.rootcovs[k,:,:]
        P=R.dot(R.T)
        return np.random.multivariate_normal(mu.reshape(-1,), P)
        
    # the neg Entropy
    def NegEntropyMC(self,nbMC=100):  
        y=0
        for i in range(0,nbMC):
            x=self.random()
            y=y+math.log(self.pdf(x))
        return y/nbMC
    
    # the left KL
    def KL(self,target,nbMC=100):  
        y=0
        for i in range(0,nbMC):
            x=self.random()
            y=y+self.logpdf(x)-target.logpdf(x)
        return y/nbMC
    
    # the left KL with fixed aleas
    def KLseed(self,target,weightSamples,normalSamples):  
        y=0
        nbMC=weightSamples.shape[0]
        
        for i in range(0,nbMC):
            k=weightSamples[i]
            mu=self.means[k,:].reshape(-1,1)
            R=self.rootcovs[k,:,:]
            u=normalSamples[i].reshape(-1,1)
            x=mu+R.dot(u)
            x=x.reshape(-1,)
            y=y+self.logpdf(x)-target.logpdf(x)
        return y/nbMC
    
    # the right KL
    def RightKL(self,target,nbMC=100):  
        y=0
        for i in range(0,nbMC):
            x=target.random()
            y=y+self.logpdf(x)-target.logpdf(x)
        return y/nbMC
    
    # the left KL with fixed aleas
    def RightKLseed(self,target,samples):  
        y=0
        nbMC=samples.shape[0]
        for i in range(0,nbMC):
            x=samples[i]
            x=x.reshape(-1,)
            y=y+self.logpdf(x)-target.logpdf(x)
        return y/nbMC
                
    # gradient of log p
    def gradient(self,x):
        y=np.zeros([self.d,1])
        for k in range(0,self.K):
            w=self.weights[k].reshape(1,)
            mu=self.means[k,:].reshape(-1,1)
            R=self.rootcovs[k,:,:].reshape(self.d,self.d)
            P=R.dot(R.T)
            e=(x-mu).reshape(-1,1)
            n=multivariate_normal.pdf(x.reshape(-1,),mu.reshape([-1,]),P)
            y=y-w*LA.inv(P).dot(e)*n
        y=y/self.pdf(x)
        #print("grad=",y)
        return y.reshape([-1,1])
                
    def plotCovs(self,ax,showBar=False,label=""):
        cmap = matplotlib.cm.get_cmap('gist_heat')
        sm = plt.cm.ScalarMappable(cmap=cmap)
        for i in range(0,self.K):
            wi=self.weights[i].reshape(1,)
            col=cmap(wi)[0]
            mui=self.means[i,:].reshape(self.d,1)
            Ri=self.rootcovs[i,:,:].reshape(self.d,self.d)
            Pi=Ri.dot(Ri.T)
            if label != "":
                graphix.plot_ellipsoid2d(ax,mui,Pi,'r',zorder=3,linestyle='-',linewidth=2,label=label)
            else:
                graphix.plot_ellipsoid2d(ax,mui,Pi,'r',zorder=3,linestyle='-',linewidth=2)
        if showBar:
            plt.colorbar(sm, ticks=np.linspace(0,1,10),boundaries=np.arange(-0.05,1.1,.1))
            
    def plot3D(self,ax,xv,yv,label="",col='r'):
        gridpdf=np.zeros((xv.shape[0],yv.shape[0]))  
        Z=0
        for i in np.arange(0,xv.shape[0]):
            for j in np.arange(0,yv.shape[0]):
                theta=np.zeros((2,1))
                theta[0]=xv[i,j]
                theta[1]=yv[i,j]
                gridpdf[i,j]=self.pdf(theta)
                Z=Z+gridpdf[i,j]
        gridpdf=gridpdf/Z
        ax.plot_wireframe(xv,yv,gridpdf,label=label,rstride=10, cstride=10,zorder = 0.5,color=col)
        
                
                
        
                                