import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from .Integration import rk4step
from .Graphix import graphix 
from .Utils import Expect
from .GMM import GMM
from abc import ABCMeta, abstractmethod
from scipy.stats import multivariate_normal

getcontext().prec = 6

class VariationalGMMprocess():
    
    def __init__(self,target,gmm0,invbeta):
        self.target=target
        self.gmm=gmm0
        self.invbeta=invbeta
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

class VGMM_Fixed(VariationalGMMprocess):
    
    def __init__(self,target,gmm0,invbeta,fmeanMethod=Expect.fmeanCKF):
        super().__init__(target,gmm0,invbeta)
        self.fmeanMethod=fmeanMethod
    
    # Propagation for a step
    def stepForward(self,dt):
        print("propag ",self.time)
        
        ### Runge Kutta integration 
        X0=self.gmm.GMM2Vector()
        Xt=rk4step(VGMM_Fixed.MixtureDynamicComplete,dt,X0,self.gmm.d,self.gmm.K,self.target,\
                   self.invbeta,self.fmeanMethod)
        ### Euler variant
        #dX=VGMM.MixtureDynamicComplete(X0,self.gmm.d,self.gmm.K,self.target,self.invbeta,self.fixedWeights)
        #Xt=X0+dt*dX
        gmm=GMM.Vector2GMM(Xt,self.gmm.d,self.gmm.K)
        return gmm
    
    def MixtureDynamicComplete(X,d,K,target,invBeta,fmeanMethod):
        gmm=GMM.Vector2GMM(X,d,K)
        dX=np.empty((0,1))             
        for k in range(0,K):
            muk=gmm.means[k,:].reshape(d,1)
            Rk=gmm.rootcovs[k,:,:].reshape(d,d)

            # compute MEAN update
            EgradLogp=fmeanMethod(target.gradient,muk,Rk)
            EhessLogpP=fmeanMethod(Expect.ExJf,muk,Rk,muk,target.gradient)
            
            EgradLogq=fmeanMethod(gmm.gradient,muk,Rk)
            EhessLogqP=fmeanMethod(Expect.ExJf,muk,Rk,muk,gmm.gradient)
            
            dmuk=invBeta*(EgradLogp-EgradLogq)
            A=EhessLogpP-EhessLogqP
            dPk=invBeta*(A+A.T)
            # compute the sqrt derivative:
            invRk=LA.inv(Rk)
            L=Expect.lower(invRk.dot(dPk).dot(invRk.T))
            dRk=Rk.dot(L)
            # put the gradient in a vector:
            dX=np.concatenate((dX,np.zeros([1,1]),dmuk.reshape(-1,1),dRk.reshape(-1,1)), axis=0)
        return dX
    
class VGMM_Fisher(VariationalGMMprocess):
    
    def __init__(self,target,gmm0,invbeta,fixedWeights=False,fmeanMethod=Expect.fmeanCKF):
        super().__init__(target,gmm0,invbeta)
        self.fixedWeights=fixedWeights
        self.fmeanMethod=fmeanMethod
    
    # Propagation for a step
    def stepForward(self,dt):
        print("propag ",self.time)
        
        ### Runge Kutta integration 
        X0=self.gmm.GMM2Vector2()
        Xt=rk4step(VGMM_Fisher.MixtureDynamicComplete,dt,X0,self.gmm.d,self.gmm.K,self.target,\
                   self.invbeta,self.fmeanMethod)
        ### Euler variant
        #dX=VGMM.MixtureDynamicComplete(X0,self.gmm.d,self.gmm.K,self.target,self.invbeta,self.fixedWeights)
        #Xt=X0+dt*dX
        gmm=GMM.Vector2GMM2(Xt,self.gmm.d,self.gmm.K)
        return gmm
    
    def MixtureDynamicComplete(X,d,K,target,invBeta,fmeanMethod):
        gmm=GMM.Vector2GMM2(X,d,K)
        dX=np.empty((0,1))          
        for k in range(0,K):
            wk=gmm.weights[k].reshape(1,)
            muk=gmm.means[k,:].reshape(d,1)
            Rk=gmm.rootcovs[k,:,:].reshape(d,d)
            # compute MEAN update
            ELogp=fmeanMethod(target.logpdf,muk,Rk)
            EgradLogp=fmeanMethod(target.gradient,muk,Rk)
            EhessLogpP=fmeanMethod(Expect.ExJf,muk,Rk,muk,target.gradient)
            
            ELogq=fmeanMethod(gmm.logpdf,muk,Rk)
            EgradLogq=fmeanMethod(gmm.gradient,muk,Rk)
            EhessLogqP=fmeanMethod(Expect.ExJf,muk,Rk,muk,gmm.gradient)
            
            dalphak=invBeta*(ELogp-ELogq)*math.sqrt(wk)
            dalphak=np.asarray(dalphak)
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
    
