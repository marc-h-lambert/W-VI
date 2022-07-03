###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational inference via Wasserstein gradient flows"                          #
# Authors: Marc Lambert, Sinho Chewi, Francis Bach,                               #
#          Silvère Bonnabel, Philippe Rigollet                                    #
###################################################################################
# Main file for tests on Gaussian approximation 
###################################################################################

import numpy as np
import os
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from Core.VariationalGP import VGP_JKO
from Core.LangevinTarget import logisticpdf, logisticpdfHD
from Core.Laplace import LaplaceLogReg
from VGA_plot import plotGPtraj, plotHistoKLgaussian
from VGA_test import TestLaplace2D, TestHD, BiModalTarget

if __name__ == "__main__":
    ########## Choose here your desired test: VGA-Bimodal, VGA-LogReg or VGA-LogReg-HD  ################

    TEST=["VGA-LogReg"]
    
    num=0
    # Wasersstein VI applied on a bimodal target in dim 2 (plot the trajectory) 
    if "VGA-Bimodal" in TEST:
        print("Test one Gaussian vs a target with two modes")
        d=2
        K=2
        step=1
        fixedWeights=False
        T=10
        invbeta=1
        FileName="VGA-BiModal-Step={}".format(step)
        TitleText="Mixture of $2$ Gaussians \n step size {}".format(step)
        
        # The target to approximate
        target=BiModalTarget()
        print("computing true posterior...")
        Npoints=100
        target.ComputeGrid2D(10,Npoints=Npoints)
        
        mean0=np.array([-8, -2])
        cov0=np.identity(2)
        vgp=VGP_JKO(target,mean0,cov0,invbeta)
        vgp.propagate(step, T)
        
        plotGPtraj(target,vgp,TitleText,FileName,num=num,step=1,coefStep=2,x1=-12,x2=10,y1=-11,y2=11)

        num=num+1
    
    # Wasersstein VI applied on a logistic target in dim 2 (plot the trajectory 
    # and Laplace and KL) 
    if "VGA-LogReg" in TEST:
        print("Test one Gaussian vs a logistic target")
        TestLaplace2D(N=10,s=1.5,radius=10,scaleCov=10,num=num,Npoints=100)
        num=num+8
        TestLaplace2D(N=10,s=2,radius=20,scaleCov=10,num=num,Npoints=200)
        num=num+8
    
    # Wasersstein VI applied on a logistic target in dim > 2 (plot the trajectory 
    # and Laplace and KL) 
    if "VGA-LogReg-HD" in TEST:
        print("Test one Gaussian vs a target from logistic reg (HD)")
        TestHD(d=10,N=50,s=5,num=num,scaleCovVI=100,Z=2e10,scale=1.2,seed=2)
        num=num+4
        #TestHD(d=10,N=50,s=10,num=num,scaleCovVI=100,Z=1e100,scale=1.2)
        #num=num+4
        # TestHD(d=100,N=500,s=5,num=num,scaleCovVI=100,scale=3)
        # num=num+4
        # TestHD(d=100,N=500,s=10,num=num,scaleCovVI=100,Z=1e100,scale=3)
        # num=num+4

 