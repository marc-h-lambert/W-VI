###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Generate N synthetic noisy observations in dimension d for :                    #
# - the linear regression problem (with Gaussian inputs and an ouput noise)       #
# - the logistic regression problem (with two Gaussian inputs for Y=0 and Y=1)    #
# The Gaussian covariance on inputs are parametrized by                           #
# c, scale, rotate and normalize                                                  #
###################################################################################

import numpy.linalg as LA 
import numpy as np
from scipy.stats import special_ortho_group
#from .Utils import sigmoid
from .Graphix import graphix 
import math
from mpl_toolkits.mplot3d import Axes3D 

def sigmoid(x):
    x=np.clip(x,-100,100)
    return 1/(1+np.exp(-x))

class LogisticRegDatasetHD():
    def __init__(self, meansShift,N,d):
        print("meansShift=",meansShift)
        self._meansShift = meansShift# the distance between the means
        self._CovInputs=np.identity(d)/math.sqrt(d)
        theta=np.random.rand(d,)
        theta=theta/LA.norm(theta)
        print("theta=",LA.norm(theta))
        self._meanInputs0 = -theta*self._meansShift/2
        self._meanInputs1 = theta*self._meansShift/2 
        
        invCov=LA.inv(self._CovInputs)
        self._thetaOpt=invCov.dot(self._meanInputs1-self._meanInputs0)
        print("normThetaOpt=",LA.norm(self._thetaOpt))
        
        # generate the inputs
        X0=np.random.multivariate_normal(self._meanInputs0,self._CovInputs,int(N/2))
        X1=np.random.multivariate_normal(self._meanInputs1,self._CovInputs,int(N/2)) 
        X=np.concatenate((X0,X1))

        Y0=np.ones((int(N/2),1))*0
        Y1=np.ones((int(N/2),1))*1
        Y=np.concatenate((Y0,Y1))
        DataSet=list(zip(Y,X))
        np.random.shuffle(DataSet)
        Y,X= zip(*DataSet)
        self._outputs,self._inputs = np.array(Y),np.array(X)
        
    def plot(self,ax,plotcov=True,plotNormal=True,scale=1):
        Y=self._outputs
        X=self._inputs
        N,d=X.shape
        
        # plot cloud point
        Y=Y.reshape(N,1)
        ax.scatter(X[np.where(Y==0)[0],0],X[np.where(Y==0)[0],1])
        ax.scatter(X[np.where(Y==1)[0],0],X[np.where(Y==1)[0],1])
    
        # plot ellipsoids 
        if plotcov:
            X0=X[np.where(Y==0)[0]]
            Cov=np.cov(X0.T)
            #graphix.plot_ellipsoid2d(ax,self._meanInputs0[0:2],Cov[0:2,0:2],linestyle='-',linewidth=2,label='Covariance after normalization')
            graphix.plot_ellipsoid2d(ax,self._meanInputs0[0:2],self._CovInputs[0:2,0:2],linestyle='-',linewidth=2)
            X1=X[np.where(Y==1)[0]]
            Cov=np.cov(X1.T)
            #graphix.plot_ellipsoid2d(ax,self._meanInputs1[0:2],Cov[0:2,0:2],linestyle='-',linewidth=2)
            graphix.plot_ellipsoid2d(ax,self._meanInputs1[0:2],self._CovInputs[0:2,0:2],linestyle='-',linewidth=2)
    
        if plotNormal:
            # plot separator and normal (optimal) --> the norm of the vector show the confidence of classification
            x=np.arange(-5*scale/math.sqrt(d),5*scale/math.sqrt(d),0.001)
            #x=np.arange(-3,3,0.01)
            y=-(self._thetaOpt[0]/self._thetaOpt[1])*x
            ax.plot(x,y,'b',label='separator',linewidth=2,markeredgewidth=0.1,markeredgecolor='bk')
            ax.arrow(0,0,self._thetaOpt[0],self._thetaOpt[1],width=0.03,length_includes_head=True, label='Theta')
        #ax.set_xlim(-3,3)
        #ax.set_ylim(-3,3)
        ax.set_xlim(-5*scale/math.sqrt(d),5*scale/math.sqrt(d))
        ax.set_ylim(-5*scale/math.sqrt(d),5*scale/math.sqrt(d))

    def plotOutputs(self, ax):
        theta=self._thetaOpt/LA.norm(self._thetaOpt)
        MU=sigmoid(self._inputs.dot(theta))
        ax.hist(MU,50)
        
    @property
    def datas(self):
        return self._outputs,self._inputs
    
    @property
    def optim(self):
        return self._thetaOpt
 
class observations(object):
    def __init__(self, N,d,c,scale,rotate,normalize,seed):
        self._N = N # the number of observations
        self._d = d # the dimension of inputs
        self._c = c # a parameter driving the condition number of covariance of inputs 
        self._scale = scale # the inputs scale (1 by default)
        self._rotate = rotate #true if the covariance of inputs are rotated
        self._normalize = normalize #true if the covariance of inputs are rotated
        self._seed = seed # the random seed (to reproduce results)
    
    @property
    def N(self):
        return self._N
    
    @property
    def d(self):
        return self._d
    
    def covariance(self,normalize=True):
        vec=(1/np.arange(1,self._d+1)**self._c)*self._scale**2
        if normalize:
            vec=vec/LA.norm(vec)**2
        Cov_u=np.diag(vec)
        if self._d>1 and self._rotate:
            np.random.seed(self._seed)
            Q = special_ortho_group.rvs(dim=self._d)
            Cov_u=np.transpose(Q).dot(Cov_u).dot(Q)
        return Cov_u
    
    @property
    def datas(self):
        pass
    
    @property
    def optim(self):
        pass
    
    @property
    def covInputs(self):
        pass
    
class LogisticRegDataset(observations):
    def __init__(self, meansShift,N,d,c,seed,scale=1,isotropic=False,normalize=False, rotate=True):
        super().__init__(N,d,c,scale,rotate,normalize,seed)
        self._meansShift = meansShift # the distance between the means
        if isotropic:
            self._CovInputs=np.identity(d)/math.sqrt(d)
        else:
            self._CovInputs=self.covariance(normalize=self._normalize)
        
        # we normalize the means
        np.random.seed(seed)
        mean_dir=np.random.rand(d,)
        theta=mean_dir/LA.norm(mean_dir)
        self._meanInputs0 = theta*self._meansShift/2
        self._meanInputs1 = -theta*self._meansShift/2 
        
        invCov=LA.inv(self._CovInputs)
        #gamma=0.5*self.__meanInputs0.T.dot(invCov).dot(self.__meanInputs0)-0.5*self.__meanInputs1.T.dot(invCov).dot(self.__meanInputs1)
        #print('gamma=(must be 0)',gamma)
        self._thetaOpt=invCov.dot(self._meanInputs1-self._meanInputs0)
    
        # generate the inputs
        np.random.seed(seed)
        X0=np.random.multivariate_normal(self._meanInputs0,self._CovInputs,int(N/2))
        #if normalize: (not used, it is equivalent to normalize Cov)
        #    X0=self.__meanInputs0+(X0-self.__meanInputs0)/LA.norm(np.std(X0,axis=0))
        np.random.seed(seed+1)
        X1=np.random.multivariate_normal(self._meanInputs1,self._CovInputs,int(N/2)) 
        #if normalize: (not used, it is equivalent to normalize Cov)
        #    X1=self.__meanInputs1+(X1-self.__meanInputs1)/LA.norm(np.std(X1,axis=0))
        X=np.concatenate((X0,X1))

        # generate the outputs
        Y0=np.ones((int(N/2),1))*0
        Y1=np.ones((int(N/2),1))*1
        Y=np.concatenate((Y0,Y1))
        DataSet=list(zip(Y,X))
        np.random.shuffle(DataSet)
        Y,X= zip(*DataSet)
        self._outputs,self._inputs = np.array(Y),np.array(X)
        
    def plot(self,ax,plotcov=True,plotNormal=True):
        Y=self._outputs
        X=self._inputs
        N,d=X.shape
        
        # plot cloud point
        Y=Y.reshape(N,1)
        ax.scatter(X[np.where(Y==0)[0],0],X[np.where(Y==0)[0],1])
        ax.scatter(X[np.where(Y==1)[0],0],X[np.where(Y==1)[0],1])
    
        # plot ellipsoids 
        if plotcov:
            X0=X[np.where(Y==0)[0]]
            Cov=np.cov(X0.T)
            #graphix.plot_ellipsoid2d(ax,self._meanInputs0[0:2],Cov[0:2,0:2],linestyle='-',linewidth=2,label='Covariance after normalization')
            graphix.plot_ellipsoid2d(ax,self.meanInputs0[0:2],self._CovInputs[0:2,0:2],linestyle='-.',linewidth=2,label='Covariance used for generation')
            X1=X[np.where(Y==1)[0]]
            Cov=np.cov(X1.T)
            #graphix.plot_ellipsoid2d(ax,self._meanInputs1[0:2],Cov[0:2,0:2],linestyle='-',linewidth=2)
            graphix.plot_ellipsoid2d(ax,self.meanInputs1[0:2],self._CovInputs[0:2,0:2],linestyle='-.',linewidth=2)
    
        if plotNormal:
            # plot separator and normal (optimal) --> the norm of the vector show the confidence of classification
            #x=np.arange(-1/math.sqrt(d),1/math.sqrt(d),0.001)
            x=np.arange(-3,3,0.001)
            y=-self._thetaOpt[0]/self._thetaOpt[1]*x
            ax.plot(x,y,'b',label='separator',linewidth=2,markeredgewidth=0.1,markeredgecolor='bk')
            ax.arrow(0,0,self._thetaOpt[0],self._thetaOpt[1],width=0.1,length_includes_head=True, label='Theta')
        ax.set_xlim(-4,4)
        ax.set_ylim(-4,4)

    def plotOutputs(self, ax):
        theta=self._thetaOpt/LA.norm(self._thetaOpt)
        MU=sigmoid(self._inputs.dot(theta))
        ax.hist(MU,50)
        
    @property
    def datas(self):
        return self._outputs,self._inputs
    
    @property
    def optim(self):
        return self._thetaOpt
    
    @property
    def covInputs(self):
        return self._CovInputs
    
    @property
    def meanInputs0(self):
        return self._meanInputs0
    
    @property
    def meanInputs1(self):
        return self._meanInputs1
    
        
    