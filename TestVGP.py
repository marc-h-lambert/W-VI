import numpy as np
import os
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from Core.VariationalGMM import VGMM, WSGD
from Core.GMM import GMM
from Core.LangevinTarget import logisticpdf, logisticpdfHD
from Core.Laplace import LaplaceLogReg
from plotHelpers import generateGMM, generateGMMrandom, generateGMMsquare,\
    generateCovariance, plotGMMcovs, plotGMMtraj, \
        recordVideo, plotHistoKL, plotHistoKLgaussian, KLnormalVslogpdf, \
            computeResults, plotHistoRightKL
from TestHelpers import *
#from scipy.stats import multivariate_normal
#from scipy.stats import special_ortho_group
#import subprocess
#path=os.getcwd()


if __name__ == "__main__":
    ########## Choose here your desired test GMM-1, GMM-2, etc. ################

    TEST=["VGA3-HD"]#"GMMvsBiModal","GMMvsLogReg"]
    
    num=0
    listK=[4,20,40]
    #listK=[4,5,6]
    if "Laplace" in TEST:
        d=2
        N=10
        target=logisticpdf(s=1.5,N=N,d=d)
        Y,X=target.Obs.datas
        Y=Y.reshape(N,)
        mean0=np.zeros([d,1])
        lap=LaplaceLogReg(mean0)
        lap.fit(X,Y)
        print(lap.mode)
        print(lap.Cov)
        fig, ax = plt.subplots(1, 1,figsize=(3,4),num=num)
        lap.plotEllipsoid(ax)
        
    if "VGA1" in TEST:
        print("Sensitivity to the initial conditions for one Gaussian")
        d=2
        K=2
        step=0.1
        fixedWeights=False
        T=50
        invbeta=1
        FileName="VGA-BiModal-Step={}".format(step)
        TitleText="mean at (-10,0)"
        # The target to approximate
        listMean=np.array([[5, 5], [5, -5]])#, [2, 1],[0, 1]]
        listCov=np.array([np.identity(2), np.identity(2)])
        listw=np.array([0.5,0.5])
        #listw=np.array([0.7,0.3])
        target=GMM(listw,np.asarray(listMean),GMM.cov2sqrt(listCov))
        
        # bad init:
        K=1
        listw0=np.ones([K,1])
        listw0=listw0/listw0.sum(axis=1)
        mean0=[-10, 0]

        listMean0=np.array([mean0])
        listCov0=[]
        for i in range(0,K):
            listCov0.append([np.identity(2)])
        
        print("computing true posterior...")
        Npoints=100
        target.ComputeGrid2D(10,Npoints=Npoints)
       
        gmm=GMM(listw0,listMean0,GMM.cov2sqrt(listCov0))
        vgmm=VGMM(target,gmm,1/invbeta,fixedWeights=True)
        vgmm.propagate(step, T)
        
        plotGMMtraj(target,vgmm,TitleText,FileName,num=num,step=1,coefStep=2)
        
    if "VGA2" in TEST:
        print("Test one Gaussian vs a target with two modes")
        d=2
        K=2
        step=1
        fixedWeights=False
        T=30
        invbeta=1
        FileName="VGA-BiModal-Step={}".format(step)
        TitleText="Mixture of $2$ Gaussians \n step size {}".format(step)
        
        # The target to approximate
        target=BiModalTarget()
        print("computing true posterior...")
        Npoints=100
        target.ComputeGrid2D(10,Npoints=Npoints)
        
        # setting GMM
        K=1
        listw0=np.ones([K,1])
        listw0=listw0/listw0.sum(axis=1)
        mean0=[-8, -2]

        listMean0=np.array([mean0])
        listCov0=[]
        for i in range(0,K):
            listCov0.append([np.identity(2)])
        
        gmm=GMM(listw0,listMean0,GMM.cov2sqrt(listCov0))
        
        # optim
        vgmm=VGMM(target,gmm,1/invbeta,fixedWeights=True)
        vgmm.propagate(step, T)
        
        plotGMMtraj(target,vgmm,TitleText,FileName,num=num,step=1,coefStep=2,x1=-12,x2=10,y1=-11,y2=11)
        ax.set_aspect("equal")
        num=num+1
                
    if "VGA3" in TEST:
        
        TestLaplace2D(N=10,s=1.5,radius=10,scaleCov=10,num=num,Npoints=100)
        num=num+8
        TestLaplace2D(N=10,s=2,radius=20,scaleCov=10,num=num,Npoints=200)
        num=num+8
        
    if "VGA3-HD" in TEST:
        print("Test one Gaussian vs a target from logistic reg (HD)")
        #TestCKF(N=150,s=5,d=30,num=num)
        #num=num+1
        # Test s=5
        TestHD(d=10,N=50,s=5,num=num,scaleCovVI=100,scaleCovLap=-1,scale=1.2)
        num=num+4
        TestHD(d=30,N=150,s=5,num=num,scaleCovVI=100,scaleCovLap=-1,scale=2,Z=1e10)
        num=num+4
        TestHD(d=100,N=500,s=5,num=num,scaleCovVI=100,scaleCovLap=-1,scale=3)
        num=num+4
        
        # Test s=10
        TestHD(d=10,N=50,s=10,num=num,scaleCovVI=100,scaleCovLap=-1,Z=1e100,scale=1.2)
        num=num+4
        TestHD(d=30,N=150,s=10,num=num,scaleCovVI=100,scaleCovLap=-1,Z=1e100,scale=2)
        num=num+4
        TestHD(d=100,N=500,s=10,num=num,scaleCovVI=100,scaleCovLap=-1,Z=1e100,scale=3)
        num=num+4
        
                
    if "GMMvsBiModal" in TEST:
        
        print("setting target...")
        target=BiModalTarget()
        Npoints=100
        Rtarget=10
        target.ComputeGrid2D(Rtarget,Npoints=Npoints)
        center=target.mean()
        x1=center[0]-10
        x2=center[0]+10
        y1=center[1]-10
        y2=center[1]+10
        K=40
        T=30
        stepRK=1
        stepPlot=int(1/stepRK)
        Name="Target1" 
        Nfigs=4
        TestSingleGaussianSampling(target,Name=Name,Nfigs=Nfigs,\
                                   K=K,T=T,Rmodel=10,num=0,center=center,\
                                       stepRK=stepRK,stepPlot=stepPlot,\
                                       x1=x1,x2=x2,y1=y1,y2=y2)
        num=num+Nfigs
        
    if "GMMvsLogReg" in TEST:
        
        print("setting target...")
        target=logisticpdf(s=1.5,N=10,d=2)
        Npoints=100
        Rtarget=10
        target.ComputeGrid2D(Rtarget,Npoints=Npoints,center=np.array([2.5,-5]))
        center=target.mean()
        x1=center[0]-10
        x2=center[0]+10
        y1=center[1]-10
        y2=center[1]+10
        K=40
        T=30
        stepRK=1
        stepPlot=int(1/stepRK)
        Name="Target2" 
        Nfigs=4
        TestSingleGaussianSampling(target,Name=Name,Nfigs=Nfigs,\
                                   K=K,T=T,Rmodel=7,num=0,center=center,\
                                       stepRK=stepRK,stepPlot=stepPlot,\
                                       x1=x1,x2=x2,y1=y1,y2=y2)
        num=num+Nfigs
        
    if "GMMTarget1" in TEST:
        print('----------------------------------------------')
        print("TEST with 4 equally weighted modes")
        print('----------------------------------------------')
        
        # The target to approximate
        listwS=[0.25,0.25,0.25,0.25]
        Rtarget=3
        target=generateGMMsquare(center=[0,0],radius=Rtarget,scaleCov=2,listw=listwS)
        
        Name="Target1" 
        TestGaussianSampling(target,Name,listK=listK,num=num)
        num=num+1+len(listK)
        
    if "GMMTarget2" in TEST:
        print('----------------------------------------------')
        print("TEST with 4 non-equally weighted modes")
        print('----------------------------------------------')
         
        # The target to approximate
        listwS=[0.1,0.4,0.2,0.3]
        Rtarget=3
        target=generateGMMsquare(center=[0,0],radius=Rtarget,scaleCov=2,listw=listwS)
        
        Name="Target2" 
        TestGaussianSampling(target,Name,listK=listK,num=num)
        num=num+1+len(listK)

    if "GMMTarget3" in TEST:
        print('----------------------------------------------')
        print("TEST with 6 non-isotropic modes")
        print('----------------------------------------------')
         
        # The target to approximate
        NModes=6
        Rtarget=5
        listw=np.array([0.2,0.1,0.2,0.2,0.05,0.15])
        target=generateGMM(NModes=NModes,listw=listw,R=Rtarget,scaleCov=1.3,randCov=True,Scaling=1)
        
        Name="Target3"
        TestGaussianSampling(target,Name,listK=listK,Rmodel=1.3*12,num=num)
        num=num+1+len(listK)

        
        
        
        
    