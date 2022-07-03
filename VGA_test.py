###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational inference via Wasserstein gradient flows"                          #
# Authors: Marc Lambert, Sinho Chewi, Francis Bach,                               #
#          Silvère Bonnabel, Philippe Rigollet                                    #
###################################################################################
###################################################################################
# Helpers functions for tests on Gaussian approximation
###################################################################################

import numpy as np
import os
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from Core.Utils import Expect
from Core.GMM import GMM
from Core.VariationalGP import VariationalGaussianProcess,VGP_JKO
from Core.LangevinTarget import logisticpdf, logisticpdfHD
from Core.Laplace import LaplaceLogReg
from plotHelpers import plotGPtraj, plotHistoKL, plotHistoKLgaussian, KLnormalVslogpdf, \
            computeResults, plotHistoRightKL
from scipy.stats import multivariate_normal
from scipy.stats import special_ortho_group


def TestLaplace2D(N,s,radius=8,scaleCov=1,num=0,Npoints=100):
    d=2
    step=0.1
    stepPlot=int(1/step)
    T=30
    invbeta=1
    TitleText="Logistic regression - step size {}".format(step)
    FileName="GaussianVI-LogReg-d{}-N{}-s{}".format(d,N,int(s))
    
    # The target to approximate
    target=logisticpdf(s=s,N=N,d=d)
    print("computing true posterior...")
    Npoints=Npoints
    center=np.array([-10,0])
    target.ComputeGrid2D(10,Npoints=Npoints,center=center)
    center=target.mean()
    target.ComputeGrid2D(radius,Npoints=Npoints,center=center)
    x1=center[0]-radius
    x2=center[0]+radius
    y1=center[1]-radius
    y2=center[1]+radius
    
    
    # run Wasserstein VI
    mean0=np.array([-10, 0])
    cov0=np.identity(2)*scaleCov
    vgp=VGP_JKO(target,mean0,cov0,invbeta)
    vgp.propagate(step, T)
        
    # run Laplace (non regularized)
    Y,X=target.Obs.datas
    Y=Y.reshape(N,)    
    lap=LaplaceLogReg(mean0)
    lap.fit(X,Y)
    
    # plot datas inputs
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    target.plotObs(ax)
    plt.savefig("imgs/{}-Datas.pdf".format(FileName))
    num=num+1
    
    # plot datas outputs
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    target.Obs.plotOutputs(ax)
    plt.savefig("imgs/{}-Histo.pdf".format(FileName))
    num=num+1
    
    # plot VI traj
    plotGPtraj(target,vgp,TitleText,FileName,num=num,step=1,coefStep=2,\
                x1=x1,x2=x2,y1=y1,y2=y2)
    #ax.set_aspect("equal")
    num=num+1
    
    # plot 2D cov lap vs VI
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    num=num+1
    vgp.plotOptimal(ax,label="Wasserstein-VI")
    target.plot(ax)
    target.plotEllipsoid(ax,col='k',label="True moments")
    lap.plotEllipsoid(ax,col='b',label="Laplace")
    lap2=LaplaceLogReg(mean0,np.identity(2)*1000)
    lap2.fit(X,Y)
    lap2.plotEllipsoid(ax,col='c',label="regularized Laplace")
    plt.legend(loc="upper left", prop={'size': 6})
    plt.xlim(x1,x2)
    plt.ylim(y1,y2)
    ax.relim()
    ax.autoscale_view()
    plt.tight_layout()
    plt.savefig("imgs/{}-2Dplot.pdf".format(FileName))
    plt.show()
    
    lapVGP=VariationalGaussianProcess(target,lap.mode,lap.Cov)
    kl_lap=lapVGP.KL(-1)
    plotHistoKLgaussian(target,[vgp],["VI"],Name=FileName,num=num,\
              stepSize=stepPlot,kl_laplace=kl_lap)
    num=num+1

def TestHD(N,s,d,scaleCovVI=1,num=0,Z=1,scale=1,seed=1):
    print("---------------- Test d={} and N={} ---------".format(d,N))
    step=0.1
    stepPlot=int(1/step)
    fixedWeights=False
    T=30
    invbeta=1
    TitleText="Logistic regression - step size {}".format(step)
    FileName="GaussianVI-LogRegHD-d{0}-N{1}-s{2}".format(d,N,int(s))
    sep=s/math.sqrt(d)
    print("sep=",sep)
    # The target to approximate
    target=logisticpdfHD(s=sep,N=N,d=d,Z=Z,seed=seed)
    
    # The initial Gaussian 
    mean0=np.zeros([d, ])
    cov0=np.identity(d)*scaleCovVI
    vgp=VGP_JKO(target,mean0,cov0,invbeta)
    vgp.propagate(step, T)
        
    # plot datas inputs
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    target.plotObs(ax,scale=scale)
    plt.savefig("imgs/{}-Datas.pdf".format(FileName))
    num=num+1
    
    # plot datas outputs
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    target.Obs.plotOutputs(ax)
    plt.savefig("imgs/{}-Histo.pdf".format(FileName))
    num=num+1
    
    #plot left KL
    if not (d==100 and s==10):
         # run Laplace (non regularized)
        Y,X=target.Obs.datas
        Y=Y.reshape(N,)   
        lap=LaplaceLogReg(mean0)
        lap.fit(X,Y)
        lapVGP=VariationalGaussianProcess(target,lap.mode,lap.Cov)
        kl_lap=lapVGP.KL(-1)
        # nbMC=1000
        # normalSamples=np.random.multivariate_normal(np.zeros(d,), np.identity(d),size=(nbMC,))
        # kl_lap=lap.KLseed(target,normalSamples)
    else:
        kl_lap=None
    #print("Left KL Laplace v2=",left_kl_lapv2)
    plotHistoKLgaussian(target,[vgp],["VI"],Name=FileName,num=num,\
              stepSize=stepPlot,kl_laplace=kl_lap)
    num=num+1

def BiModalTarget():
    listMean=np.array([[5, 3.3], [5, -3.3]])
    listCov=np.array([np.diag([3,1]), np.diag([1,3])])
    listw=np.array([0.5,0.5])
    target=GMM(listw,np.asarray(listMean),GMM.cov2sqrt(listCov))
    return target


