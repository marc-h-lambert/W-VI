import numpy as np
import matplotlib
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

# A structure to store the parameters of a GMM
class GMM():
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
    
    # the neg Entropy
    def KL(self,target,nbMC=100):  
        y=0
        for i in range(0,nbMC):
            x=self.random()
            y=y+math.log(self.pdf(x))-math.log(target.pdf(x))
        return y/nbMC
                
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
                
    def plotCovs(self,ax,showBar=False):
        cmap = matplotlib.cm.get_cmap('gist_heat')
        sm = plt.cm.ScalarMappable(cmap=cmap)
        for i in range(0,self.K):
            wi=self.weights[i].reshape(1,)
            col=cmap(wi)[0]
            mui=self.means[i,:].reshape(self.d,1)
            Ri=self.rootcovs[i,:,:].reshape(self.d,self.d)
            Pi=Ri.dot(Ri.T)
            graphix.plot_ellipsoid2d(ax,mui,Pi,'r',zorder=3,linestyle='-',linewidth=2)
        if showBar:
            plt.colorbar(sm, ticks=np.linspace(0,1,10),boundaries=np.arange(-0.05,1.1,.1))
                                
class VariationalGMMprocess():
    
    def __init__(self,target,gmm0,beta):
        self.target=target
        self.gmm=gmm0
        self.invbeta=1/beta
        self.traj_gmm=[]
        self.traj_gmm.append(self.gmm)
        self.time=0
    
    def propagate(self,dt,T):
        while self.time < T:    
            #print(self.time)
            self.time=self.time+Decimal(dt)
            
            # Kalman propagation step
            self.gmm=self.stepForward(dt)
            
            self.traj_gmm.append(self.gmm)
        return self.traj_gmm
    
    @abstractmethod
    # Propagation for a step
    def stepForward(self,dt):
        return 

class VGMM(VariationalGMMprocess):
    
    def __init__(self,target,gmm0,beta,fixedWeights=False):
        super().__init__(target,gmm0,beta)
        self.fixedWeights=fixedWeights
    
    # Propagation for a step
    def stepForward(self,dt):
        print("propag ",self.time)
        
        ### Runge Kutta integration 
        X0=self.gmm.GMM2Vector()
        Xt=rk4step(VGMM.MixtureDynamicComplete,dt,X0,self.gmm.d,self.gmm.K,self.target,self.invbeta,self.fixedWeights)
        ### Euler variant
        #dX=VGMM.MixtureDynamicComplete(X0,self.gmm.d,self.gmm.K,self.target,self.invbeta,self.fixedWeights)
        #Xt=X0+dt*dX
        gmm=GMM.Vector2GMM(Xt,self.gmm.d,self.gmm.K)
        return gmm
    
    def MixtureDynamicComplete(X,d,K,target,invBeta,fixedWeights):
        gmm=GMM.Vector2GMM(X,d,K)
        dX=np.empty((0,1))             
        for k in range(0,K):
            wk=gmm.weights[k].reshape(1,)
            muk=gmm.means[k,:].reshape(d,1)
            Rk=gmm.rootcovs[k,:,:].reshape(d,d)
            Pk=Rk.dot(Rk.T)
            invPk=LA.inv(Pk)
            # compute MEAN update
            ELogp=Expect.fmeanSQRT(target.pdf,muk,Rk)
            EgradLogp=Expect.fmeanSQRT(target.gradient,muk,Rk)
            EhessLogpP=Expect.fmeanSQRT(Expect.ExJf,muk,Rk,muk,target.gradient)
            
            ELogq=Expect.fmeanSQRT(gmm.pdf,muk,Rk)
            EgradLogq=Expect.fmeanSQRT(gmm.gradient,muk,Rk)
            EhessLogqP=Expect.fmeanSQRT(Expect.ExJf,muk,Rk,muk,gmm.gradient)
            
            dalphak=invBeta*(ELogp-ELogq)/wk
            if fixedWeights:
                dalphak=dalphak*0
            dmuk=invBeta*(EgradLogp-EgradLogq)
            A=EhessLogpP-EhessLogqP
            dPk=invBeta*(A+A.T)
            # compute the sqrt derivative:
            invRk=LA.inv(Rk)
            L=Expect.lower(invRk.dot(dPk).dot(invRk.T))
            dRk=Rk.dot(L)
            # put the gradient in a vector:
            dX=np.concatenate((dX,dalphak.reshape(-1,1),dmuk.reshape(-1,1),dRk.reshape(-1,1)), axis=0)
        return dX
    
# Euler version with a stochastic gradient
class WSGD(VariationalGMMprocess):
    
    def __init__(self,target,gmm0,beta,fixedWeights=False):
        super().__init__(target,gmm0,beta)
        self.fixedWeights=fixedWeights
    
    # Propagation for a step
    def stepForward(self,dt):
        print("propag ",self.time)
        
        ### Runge Kutta integration 
        X0=self.gmm.GMM2Vector()
        dX=WSGD.MixtureDynamicComplete(X0,self.gmm.d,self.gmm.K,self.target,self.invbeta,self.fixedWeights)
        Xt=X0+dt*dX
        gmm=GMM.Vector2GMM(Xt,self.gmm.d,self.gmm.K)
        return gmm
    
    def MixtureDynamicComplete(X,d,K,target,invBeta,fixedWeights):
        gmm=GMM.Vector2GMM(X,d,K)
        dX=np.empty((0,1))             
        for k in range(0,K):
            wk=gmm.weights[k].reshape(1,)
            muk=gmm.means[k,:].reshape(d,1)
            Rk=gmm.rootcovs[k,:,:].reshape(d,d)
            Pk=Rk.dot(Rk.T)
            invPk=LA.inv(Pk)
            # compute MEAN update
            Nsamples=1
            ELogp=Expect.fmeanMC(target.pdf,muk,Rk,Nsamples)
            EgradLogp=Expect.fmeanMC(target.gradient,muk,Rk,Nsamples)
            EhessLogpP=Expect.fmeanMC(Expect.ExJf,muk,Rk,Nsamples,muk,target.gradient)
            
            ELogq=Expect.fmeanMC(gmm.pdf,muk,Rk,Nsamples)
            EgradLogq=Expect.fmeanMC(gmm.gradient,muk,Rk,Nsamples)
            EhessLogqP=Expect.fmeanMC(Expect.ExJf,muk,Rk,Nsamples,muk,gmm.gradient)
            
            dalphak=invBeta*(ELogp-ELogq)/wk
            if fixedWeights:
                dalphak=dalphak*0
            dmuk=invBeta*(EgradLogp-EgradLogq)
            A=EhessLogpP-EhessLogqP
            dPk=invBeta*(A+A.T)
            # compute the sqrt derivative:
            invRk=LA.inv(Rk)
            L=Expect.lower(invRk.dot(dPk).dot(invRk.T))
            dRk=Rk.dot(L)
            # put the gradient in a vector:
            dX=np.concatenate((dX,dalphak.reshape(-1,1),dmuk.reshape(-1,1),dRk.reshape(-1,1)), axis=0)
        return dX