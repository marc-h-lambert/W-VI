###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational inference via Wasserstein gradient flows"                          #
# Authors: Marc Lambert, Sinho Chewi, Francis Bach,                               #
#          Silvère Bonnabel, Philippe Rigollet                                    #
###################################################################################
# Main file for tests on Mixture of Gaussian approximation 
###################################################################################

import numpy as np
import os
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from Core.VariationalGMM import VGMM_Fixed
from Core.GMM import GMM
from Core.LangevinTarget import logisticpdf, logisticpdfHD
from Core.Laplace import LaplaceLogReg
from GMM_plot import generateGMM, generateGMMrandom, generateGMMsquare,\
    generateCovariance, plotGMMcovs, recordVideo, plotHistoKL, \
    plotHistoKLgaussian, KLnormalVslogpdf, computeResults
from GMM_test import TestGaussianSampling, TestGaussianSamplingSimple
from VGA_test import BiModalTarget

if __name__ == "__main__":
    ########## Choose here your desired test GMM-1, GMM-2, etc. ################

    TEST=["GMMTarget1"]
    
    num=0
    
    if "GMMvsBiModal" in TEST:
        print('----------------------------------------------')
        print("TEST Approximation of a bimodal distribution with a mixture of Gaussians")
        print('----------------------------------------------')
         
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
        K=20
        T=30
        stepRK=0.1
        stepPlot=int(1/stepRK)
        Name="Target1" 
        Nfigs=4
        TestGaussianSamplingSimple(target,Name=Name,Nfigs=Nfigs,\
                                   K=K,T=T,Rmodel=10,num=0,center=center,\
                                       stepRK=stepRK,stepPlot=stepPlot,\
                                       x1=x1,x2=x2,y1=y1,y2=y2)
        num=num+Nfigs
        
    if "GMMvsLogReg" in TEST:
        print('----------------------------------------------')
        print("TEST Approximation of a logistic distribution with a mixture of Gaussians")
        print('----------------------------------------------')
        print("setting target...")
        target=logisticpdf(s=1.5,N=10,d=2)
        Npoints=100
        Rtarget=1
        target.ComputeGrid2D(Rtarget,Npoints=Npoints,center=np.array([2.5,-5]))
        center=target.mean()
        x1=center[0]-10
        x2=center[0]+10
        y1=center[1]-10
        y2=center[1]+10
        K=20
        T=30
        stepRK=0.1
        stepPlot=int(1/stepRK)
        Name="Target2" 
        Nfigs=4
        TestGaussianSamplingSimple(target,Name=Name,Nfigs=Nfigs,\
                                   K=K,T=T,Rmodel=7,num=0,center=center,\
                                       stepRK=stepRK,stepPlot=stepPlot,\
                                       x1=x1,x2=x2,y1=y1,y2=y2)
        num=num+Nfigs
        
    if "GMMTarget1" in TEST:
        print('----------------------------------------------')
        print("TEST with 4 equally weighted modes")
        print('----------------------------------------------')
        
        # The target to approximate
        listK=[20]
        listwS=[0.25,0.25,0.25,0.25]
        Rtarget=3
        target=generateGMMsquare(center=[0,0],radius=Rtarget,scaleCov=2,listw=listwS)
        
        Name="Target1" 
        TestGaussianSampling(target,Name,listK=listK,num=num,R=10)
        num=num+1+len(listK)
        
    if "GMMTarget2" in TEST:
        print('----------------------------------------------')
        print("TEST with 4 non-equally weighted modes")
        print('----------------------------------------------')
         
        # The target to approximate
        listK=[4,20,40]
        listwS=[0.1,0.4,0.2,0.3]
        Rtarget=3
        target=generateGMMsquare(center=[0,0],radius=Rtarget,scaleCov=2,listw=listwS)
        
        Name="Target2" 
        TestGaussianSampling(target,Name,listK=listK,num=num,R=10)
        num=num+1+len(listK)

    if "GMMTarget3" in TEST:
        print('----------------------------------------------')
        print("TEST with 6 non-isotropic modes")
        print('----------------------------------------------')
         
        # The target to approximate
        NModes=6
        Rtarget=5
        listK=[4,20,40]
        listw=np.array([0.2,0.1,0.2,0.2,0.05,0.15])
        target=generateGMM(NModes=NModes,listw=listw,R=Rtarget,scaleCov=1.3,randCov=True,Scaling=1)
        
        Name="Target3"
        TestGaussianSampling(target,Name,listK=listK,R=1.3*12,num=num)
        num=num+1+len(listK)

        
        
        
        
    