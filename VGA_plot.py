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
# Helpers functions for plotting results 
###################################################################################

import numpy as np
import os
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
#from Core.TargetDistribution import GaussianMixture
from Core.GMM import GMM
from Core.LangevinTarget import logisticpdf
from scipy.stats import multivariate_normal
from scipy.stats import special_ortho_group
from Core.Utils import KLnormalVslogpdf
import subprocess
from matplotlib.ticker import MaxNLocator

path=os.getcwd()


def plotGPtraj(target,vgp,TitleText,FileName,num=0,step=2,coefStep=1,x1=-15,x2=15,y1=-15,y2=15):
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    target.plot(ax)
    T=len(vgp.traj_mean)
    print("T=",T)
    idx=0
    while idx < T:
        vgp.plot(ax,idx)
        idx=idx+step
        step=int(step*coefStep)
    vgp.plot(ax,T-1)
    #plt.suptitle(TitleText)
    plt.xlim([x1, x2])
    plt.ylim([y1, y2])
    #plt.axis('off')
    plt.xticks([], [])
    plt.yticks([], [])
    #ax.set_aspect("equal")
    #fig.set_size_inches(5, 5)
    plt.tight_layout()
    plt.savefig("imgs/{}-Traj.pdf".format(FileName))
    plt.show() 
    

def plotHistoKLgaussian(target,listvgp,listLabels,Name,num=0,stepSize=1,\
                        kl_laplace=None,text=""):
    fig, ax = plt.subplots(1, 1,figsize=(3.5,3),num=num)
    #plt.yscale("log")
    os.chdir(path)
    col=['r','g','c','m']
    for i in range(0,len(listvgp)):
        vgp=listvgp[i]
        label=listLabels[i]
        T=len(listvgp[i].traj_mean)
        histo_kl=[]
        timeIdx=np.arange(0,T-stepSize,stepSize)
        for t in timeIdx:
            kl=vgp.KL(t) 
            histo_kl.append(kl)
 
        plt.semilogy(timeIdx,np.array(histo_kl),color=col[i],label=label,   # data
                 linestyle='-',            # line style will be dash line
                 linewidth=2)          # line width
        #plt.grid()
        #print("histo_kl=",histo_kl)
        
    if i>0:
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
    
    
    if not kl_laplace is None:
        histo_kl=np.ones([len(timeIdx),])*kl_laplace
        plt.semilogy(timeIdx,histo_kl,color='b',label="Laplace",   # data
                 linestyle='-',            # line style will be dash line
                 linewidth=2)          # line width
        plt.legend(fontsize=12)
    #plt.ylim(-1,5)
    fontsize=15
    #plt.xticks([0,20,40,60,80,100])
    plt.xticks([0,100,200,300])
    #plt.yticks([0,1,1.5,2,2.5,3])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("step",fontsize=fontsize)
    plt.ylabel("KL divergence",fontsize=fontsize)
    plt.title("Evolution of the free energy \n"+text,fontsize=fontsize)
    
    fig.tight_layout()
    plt.savefig("imgs/{}-KL.pdf".format(Name))
    plt.show() 
    
