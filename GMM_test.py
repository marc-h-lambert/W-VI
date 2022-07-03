import numpy as np
import os
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
#from Core.TargetDistribution import GaussianMixture
from Core.VariationalGMM import VGMM_Fixed
from Core.GMM import GMM
from Core.Utils import Expect
from Core.LangevinTarget import logisticpdf, logisticpdfHD
from Core.Laplace import LaplaceLogReg
from plotHelpers import generateGMM, generateGMMrandom, generateGMMsquare,\
    generateCovariance, plotGMMcovs,plotGPtraj, \
        recordVideo, plotHistoKL, plotHistoKLgaussian, KLnormalVslogpdf, \
            computeResults, plotHistoRightKL
from scipy.stats import multivariate_normal
from scipy.stats import special_ortho_group

def TestGaussianSamplingSimple(target,Name,K,T=10,center=[],Nfigs=4,Rmodel=12,num=0,\
                               stepRK=1,stepPlot=1,x1=-15,x2=15,y1=-15,y2=15):
    d=2
    seed=2
    T=T
    stepRK=stepRK
    invbeta=1

    Rmodel=Rmodel
    
    np.random.seed(seed)
    if len(center)==0:
        center=target.mean()

    gmm=generateGMMrandom(Nsamples=K,center=center,radius=Rmodel,\
                              scaleCov=1,edgeWidth=1)
    vgmm=VGMM_Fixed(target,gmm,1/invbeta)
    print("Propagate")
    vgmm.propagate(stepRK, T)
    
    plotGMMcovs(target,vgmm,Name,center=center,num=0,Ncols=Nfigs,R=Rmodel,\
                stepSize=stepPlot,\
                x1=x1,x2=x2,y1=y1,y2=y2)
        
def TestGaussianSampling(target,Name,listK=[6],R=12,num=0):
    d=2
    seed=1
    T=30
    stepRK=1
    invbeta=1
    stepPlot=int(1/stepRK)
    #setting the initial Gaussians
    listK=listK
    listvgmm=[]
    listlabels=[]

    for K in listK:
        np.random.seed(seed)
        center=target.mean()
        gmm=generateGMMrandom(Nsamples=K,center=center,radius=12,\
                              scaleCov=1,edgeWidth=1)
        vgmm=VGMM_Fixed(target,gmm,1/invbeta)
        label="{}-{}-particles".format(Name,K)
        print("Propagate-",label)
        vgmm.propagate(stepRK, T)
        listvgmm.append(vgmm)
        listlabels.append(label)
    
    print("computing true posterior...")
    Npoints=100
    target.ComputeGrid2D(2.5*R,Npoints)
    for i in range(0,len(listvgmm)):
        computeResults(listvgmm[i],target,Name=listlabels[i],R=2.5*R,\
                    computePost=False,plot2D=True,plotKL=False,video=False,\
                        num=num,stepSize=stepPlot)
        num=num+1

    print("computing KL...")
    num=num+1
    nbMC=1000
    listnormalSamples=[]
    listweightSamples=[]
    for i in range(0,len(listvgmm)):
        gmm=listvgmm[i].gmm
        normalSamples=np.random.multivariate_normal(np.zeros(2,), np.identity(2),size=(nbMC,))
        listnormalSamples.append(normalSamples)
        weightSamples=np.random.choice(np.arange(0,gmm.K), size=(nbMC,), p=gmm.weights.reshape(-1,))
        listweightSamples.append(weightSamples)
        
    plotHistoKL(target,listvgmm,listlabels,Name=Name,num=num,listweightSamples=listweightSamples,\
                listnormalSamples=listnormalSamples,stepSize=stepPlot)
    

        



