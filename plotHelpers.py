import numpy as np
import os
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
#from Core.TargetDistribution import GaussianMixture
from Core.VariationalGMM import VGMM, WSGD
from Core.GMM import GMM
from Core.LangevinTarget import logisticpdf
from scipy.stats import multivariate_normal
from scipy.stats import special_ortho_group
from Core.Utils import KLnormalVslogpdf
import subprocess
from matplotlib.ticker import MaxNLocator

path=os.getcwd()

def generateGMM(NModes=6,listw=[],R=5,scaleCov=1,randCov=False,center=[],theta0=0):
    delta=2*math.pi/NModes
    listMean=[]
    listCov=[]
    theta=theta0
    for i in range(0,NModes):
        if len(center)==0: # Gaussian sampled on a circle 
            listMean.append([R*math.cos(theta),R*math.sin(theta)])
            theta=theta+delta 
        else: # Gaussian sampled near a center
            m=np.array(center)+np.random.uniform(0,1,2)
            listMean.append(m)
             
        diag=np.random.uniform(1,10,2)
        if randCov:
            cov=generateCovariance(diag)
        else:
            cov=np.identity(2)
        cov=cov*scaleCov
        listCov.append(cov)
          
    listMean=np.asarray(listMean)
    if len(listw) == 0:
        listw=np.ones([NModes,1])/NModes
    else:
        listw=np.array(listw)
    return GMM(listw,listMean,GMM.cov2sqrt(listCov))

def generateGMMrandom(Nsamples,center,radius,scaleCov=1,edgeWidth=-1):
    listMean=[]
    listCov=[]
    listw=np.ones([Nsamples,1])/Nsamples
   
    for i in range(0,Nsamples):
        if edgeWidth>0:#Disk
            U1 = np.random.uniform(0,1,size = 1)
            U2 = np.random.uniform(1-edgeWidth,1,size = 1)
            X = center[0] + radius * np.sqrt(U2) * np.cos(2 * np.pi * U1)
            Y = center[1] + radius * np.sqrt(U2) * np.sin(2 * np.pi * U1)
        else:
            U1 = np.random.uniform(size = 1)
            U2 = np.random.uniform(size = 1)
            X = center[0] + radius * np.sqrt(U2) * np.cos(2 * np.pi * U1)
            Y = center[1] + radius * np.sqrt(U2) * np.sin(2 * np.pi * U1)
        
        mean=np.array([X,Y]).reshape(-1,)
        listMean.append(mean)
        listCov.append(np.identity(2)*scaleCov)
    return GMM(listw,listMean,GMM.cov2sqrt(listCov))

def generateCovariance(diag,rotate=True):
    Cov_u=np.diag(diag)
    if rotate:
        Q = special_ortho_group.rvs(dim=diag.shape[0])
        Cov_u=np.transpose(Q).dot(Cov_u).dot(Q)
    return Cov_u

def generateGMMsquare(center=[0,0],radius=5,scaleCov=1,listw=np.ones([4,1])/4):
    listw=listw
    listMean=[]
    listMean.append(np.array([center[0]+radius,center[1]+radius]))
    listMean.append(np.array([center[0]+radius,center[1]-radius]))
    listMean.append(np.array([center[0]-radius,center[1]+radius]))
    listMean.append(np.array([center[0]-radius,center[1]-radius]))
    listCov=[]
    for i in range(0,4):
        listCov.append(np.identity(2)*scaleCov)
    return GMM(listw,listMean,GMM.cov2sqrt(listCov))

    
def plotGMMtraj(target,vgmm,TitleText,FileName,num=0,step=2,coefStep=1,x1=-15,x2=15,y1=-15,y2=15):
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    target.plot(ax)
    T=len(vgmm.traj_gmm)
    idx=0
    while idx < T:
        vgmm.traj_gmm[idx].plotCovs(ax)
        idx=idx+step
        step=int(step*coefStep)
    vgmm.traj_gmm[-1].plotCovs(ax)
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
    
def plotGMMcovs(target,vgmm,Name,center=[],num=0,Ncols=4,R=10,adaptStep=False,stepSize=1,\
                x1=-15,x2=15,y1=-15,y2=15):
    step=0
    showBar=False
    os.chdir(path)
    if len(center)==0:
        center=target.means.sum(axis=0)
    fig, axs = plt.subplots(1, Ncols,figsize=(3*Ncols,4),num=num)
    num=num+1
        
    for i in range(0,Ncols):
        ax=axs[i] # the axes for multi columns presentation
        figi, axi = plt.subplots(1, 1,figsize=(3,4),num=num)
        # axi: single image without title
        num=num+1
        if i+1==Ncols:
            step=len(vgmm.traj_gmm)-1
            #showBar=True
            gmm=vgmm.traj_gmm[step]
            gmm.ComputeGrid2D(R,100,center=center)
            gmm.plot(ax)
            gmm.plot(axi)
            ax.set_title("step N°{} \n (Approximated density)".format(step),fontsize=16)
        else:
            vgmm.traj_gmm[step].plotCovs(ax,showBar)
            vgmm.traj_gmm[step].plotCovs(axi,showBar)
            ax.set_title("step N°{}".format(step),fontsize=16)
            target.plot(ax)
            target.plot(axi)
        if adaptStep:
            step=step+vgmm.traj_gmm[0].K
        else:
            step=step+stepSize
            
        # Plot in axes
        ax.set_xlim([x1, x2])
        ax.set_ylim([y1, y2])
        axi.set_xlim([x1, x2])
        axi.set_ylim([y1, y2])
        ax.set_aspect("equal")
        axi.set_aspect("equal")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()
        figi.savefig("imgs/{}-2D-{}.pdf".format(Name,i))
    fig.savefig("imgs/{}-2D-{}-Synthesis.pdf".format(Name,i))
    plt.show() 
        
    
    
def plotGMMcovsOld(target,vgmm,Name,num=0,Ncols=4,R=10,adaptStep=False,stepSize=1):
    fig, axs = plt.subplots(1, Ncols,figsize=(Ncols*3,4),num=num)
    step=0
    showBar=False
    os.chdir(path)
    center=target.means.sum(axis=0)
    for i in range(0,Ncols):
        ax=axs[i]
        if i+1==Ncols:
            step=len(vgmm.traj_gmm)-1
            #showBar=True
            gmm=vgmm.traj_gmm[step]
            gmm.ComputeGrid2D(center,R,100)
            gmm.plot(ax)
            ax.set_title("step N°{} \n (Approximated density)".format(step),fontsize=16)
        else:
            vgmm.traj_gmm[step].plotCovs(ax,showBar)
            ax.set_title("step N°{}".format(step),fontsize=16)
            target.plot(ax)
        if adaptStep:
            step=step+vgmm.traj_gmm[0].K
        else:
            step=step+stepSize
    #cmap = mpl.cm.get_cmap('gist_heat')
    #sm = plt.cm.ScalarMappable(cmap=cmap)
    #plt.colorbar(sm, ticks=np.linspace(0,1,10),boundaries=np.arange(-0.05,1.1,.1))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.suptitle("Gaussian-Mixture - {}".format(Name),fontsize=20)
    plt.tight_layout()
    plt.savefig("imgs/{}-2D.pdf".format(Name))
    plt.show() 
    

    
def recordVideo(target,vgmm,Name,ContourLines=True,adaptStep=False,stepSize=1):
    T=len(vgmm.traj_gmm)
    idx=0
    os.chdir(path+'/videos/')
    center=target.means.sum(axis=0)
    R=15
    if adaptStep:
        step=vgmm.traj_gmm[0].K
    else:
        step=stepSize
    for t in np.arange(0,T,step):
        fig, (ax) = plt.subplots(1, 1,figsize=(10,5),num=0)
        gmm=vgmm.traj_gmm[t]
        if ContourLines==False:
            target.plot(ax)
            gmm.plotCovs(ax)
        else:
            gmm.ComputeGrid2D(center,R,Npoints=100)
            gmm.plot(ax)
        ax.set_aspect('equal')
        plt.savefig("GMM-{0:02d}.png".format(idx))
        plt.clf()
        idx=idx+1
    command=['/usr/local/bin/ffmpeg', '-framerate', '10', '-i', 'GMM-%02d.png', '-c:v','libx264','-pix_fmt', 'yuv420p','-y','{}.mp4'.format(Name)]
    subprocess.call(command)
    idx=0
    for t in np.arange(0,T,step):
        os.remove('GMM-{0:02d}.png'.format(idx))
        idx=idx+1
    
def plotHistoKL(target,listvgmm,listLabels,Name,num=0,listweightSamples=[],\
                listnormalSamples=[],adaptStep=False,stepSize=1,kl_laplace=None):
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    step=0
    #plt.yscale("log")
    os.chdir(path)
    col=['r','g','c','m']
    for i in range(0,len(listvgmm)):
        vgmm=listvgmm[i]
        label=listLabels[i]
        T=len(listvgmm[i].traj_gmm)
        histo_kl=[]
        if adaptStep:
            step=vgmm.traj_gmm[0].K
        else:
            step=stepSize
        for t in range(0,T,step):
            gmm=vgmm.traj_gmm[t]
            if len(listnormalSamples)>0 and len(listweightSamples)>0:
                histo_kl.append(gmm.KLseed(target,weightSamples=listweightSamples[i],\
                                           normalSamples=listnormalSamples[i]))
            else:
                histo_kl.append(gmm.KL(target,nbMC=1000))
        plt.semilogy(range(0,T,step),np.array(histo_kl),color=col[i],label=label,   # data
                 linestyle='-',            # line style will be dash line
                 linewidth=3)          # line width
        #plt.grid()
        print("histo_kl=",histo_kl)
        
    if i>0:
        ax.legend(loc="upper right")
    
    if not kl_laplace is None:
        timeIdx=range(0,T,step)
        histo_kl=np.ones([len(timeIdx),])*kl_laplace
        plt.semilogy(timeIdx,histo_kl,color='b',label="Laplace",   # data
                 linestyle='-',            # line style will be dash line
                 linewidth=2)          # line width
        plt.legend()
    #plt.ylim(-1,5)
    plt.xlabel("step")#,fontsize=10)
    plt.ylabel("Left KL divergence")#,fontsize=10)
    plt.title("Evolution of the free energy \n")#,fontsize=10)
    plt.tight_layout()
    plt.savefig("imgs/{}-KLmc.pdf".format(Name))
    plt.show() 
    
def plotHistoRightKL(target,listvgmm,listLabels,Name,num=0,listSamples=[],\
                adaptStep=False,stepSize=1,kl_laplace=None):
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    step=0
    #plt.yscale("log")
    os.chdir(path)
    col=['r','g','c','m']
    for i in range(0,len(listvgmm)):
        vgmm=listvgmm[i]
        label=listLabels[i]
        T=len(listvgmm[i].traj_gmm)
        histo_kl=[]
        if adaptStep:
            step=vgmm.traj_gmm[0].K
        else:
            step=stepSize
        for t in range(0,T,step):
            gmm=vgmm.traj_gmm[t]
            if len(listSamples)>0:
                kl=gmm.RightKLseed(target,samples=listSamples[i])
                #print(kl)
                histo_kl.append(kl)
            else:
                histo_kl.append(gmm.RightKL(target,nbMC=1000))
        plt.plot(range(0,T,step),np.array(histo_kl),color=col[i],label=label,   # data
                 linestyle='-',            # line style will be dash line
                 linewidth=2)          # line width
        plt.grid()
        
    if i>0:
        ax.legend(loc="upper right")
    
    if not kl_laplace is None:
        timeIdx=range(0,T,step)
        histo_kl=np.ones([len(timeIdx),])*kl_laplace
        plt.plot(timeIdx,histo_kl,color='b',label="Laplace",   # data
                 linestyle='-',            # line style will be dash line
                 linewidth=2)          # line width
        plt.legend()
    #plt.ylim(-1,5)
    plt.xlabel("step",fontsize=20)
    plt.ylabel("Right KL divergence",fontsize=20)
    plt.title("Evolution of the divergence \n on moments",fontsize=20)
    plt.tight_layout()
    plt.savefig("imgs/{}-KLright.pdf".format(Name))
    plt.show() 

def plotHistoKLgaussian(target,listvgmm,listLabels,Name,num=0,stepSize=1,\
                        kl_laplace=None,text=""):
    fig, ax = plt.subplots(1, 1,figsize=(3.5,3),num=num)
    #plt.yscale("log")
    os.chdir(path)
    col=['r','g','c','m']
    for i in range(0,len(listvgmm)):
        vgmm=listvgmm[i]
        label=listLabels[i]
        T=len(listvgmm[i].traj_gmm)
        histo_kl=[]
        timeIdx=np.arange(0,T-stepSize,stepSize)
        for t in timeIdx:
            gmm=vgmm.traj_gmm[t]
            kl=KLnormalVslogpdf(gmm.means[0].reshape(-1,1),gmm.rootcovs[0],target.logpdf)
            histo_kl.append(kl)
 
        plt.semilogy(timeIdx,np.array(histo_kl),color=col[i],label=label,   # data
                 linestyle='-',            # line style will be dash line
                 linewidth=2)          # line width
        #plt.grid()
        print("histo_kl=",histo_kl)
        
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
    
def computeResults(vgmm,target,Name,R=10,computePost=True,plotText=False,plot2D=False,plotKL=False,\
                   video=False,num=0,ContourLines=True,adaptStep=False,stepSize=1):
    
    if plotText:
        for t in range(0,len(vgmm.traj_gmm)):
            gmm=vgmm.traj_gmm[t]
            print("GMM at time",t)
            print("weights=",gmm.weights)
            print("means=",gmm.means)
        
    if plot2D:
        if computePost:
            print("computing true posterior...")
            center=np.asarray(target.means).sum(axis=0)/2
            Npoints=100
            target.ComputeGrid2D(center,R,Npoints)
        plotGMMcovs(target,vgmm,Name,num=num,R=R,adaptStep=adaptStep,stepSize=stepSize)
        num=num+1
        
    if plotKL:
        print("computing divergence...")
        plotHistoKL(target,[vgmm],["gmm"],Name,num=num,adaptStep=adaptStep,stepSize=stepSize)
        num=num+1
    
    if video:
        print("computing video...")
        recordVideo(target,vgmm,Name,ContourLines=ContourLines,adaptStep=adaptStep,stepSize=stepSize)