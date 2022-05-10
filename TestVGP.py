import numpy as np
import os
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
#from Core.TargetDistribution import GaussianMixture
from Core.VariationalGMM import GMM, VGMM, WSGD
from scipy.stats import multivariate_normal
from scipy.stats import special_ortho_group
import subprocess

path=os.getcwd()

def generateGMM(NModes=6,listw=[],R=5,scaleCov=1,randCov=False,center=[]):
    delta=2*math.pi/NModes
    listMean=[]
    listCov=[]
    theta=0
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
    np.random.seed(1)
    if edgeWidth>0:#Disk
        U1 = np.random.uniform(0,1,size = Nsamples)
        U2 = np.random.uniform(1-edgeWidth,1,size = Nsamples)
        X = center[0] + radius * np.sqrt(U2) * np.cos(2 * np.pi * U1)
        Y = center[1] + radius * np.sqrt(U2) * np.sin(2 * np.pi * U1)
    else:
        U1 = np.random.uniform(size = Nsamples)
        U2 = np.random.uniform(size = Nsamples)
        X = center[0] + radius * np.sqrt(U2) * np.cos(2 * np.pi * U1)
        Y = center[1] + radius * np.sqrt(U2) * np.sin(2 * np.pi * U1)
    
    for i in range(0,Nsamples):
        mean=np.array([X[i],Y[i]])
        listMean.append(mean)
        listCov.append(np.identity(2)*scaleCov)
    return GMM(listw,listMean,GMM.cov2sqrt(listCov))

def generateCovariance(diag,rotate=True):
    Cov_u=np.diag(diag)
    if rotate:
        Q = special_ortho_group.rvs(dim=diag.shape[0])
        Cov_u=np.transpose(Q).dot(Cov_u).dot(Q)
    return Cov_u

def plotGMMcovs(target,vgmm,Name,num=0,Nlines=2,Ncols=2):
    fig, axs = plt.subplots(Nlines, Ncols,figsize=(Nlines*3,Ncols*3),num=num)
    step=0
    showBar=False
    os.chdir(path)
    for i in range(0,Nlines):
        for j in range(0,Ncols):
            ax=axs[i,j]
            target.plot(ax)
            if (i+1)*(j+1)==Nlines*Ncols:
                step=len(vgmm.traj_gmm)-1
                showBar=True
            vgmm.traj_gmm[step].plotCovs(ax,showBar)
            ax.set_title("step N°{}".format(step))
            step=step+1
    #cmap = mpl.cm.get_cmap('gist_heat')
    #sm = plt.cm.ScalarMappable(cmap=cmap)
    #plt.colorbar(sm, ticks=np.linspace(0,1,10),boundaries=np.arange(-0.05,1.1,.1))
    plt.suptitle("Gaussian-Mixture - {}".format(Name))
    plt.savefig("imgs/{}-2D.pdf".format(Name))
    plt.show() 
    
def plotGMMtraj(target,vgmm,Name,num=0,step=2):
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    target.plot(ax)
    T=len(vgmm.traj_gmm)
    print(T)
    for i in range(0,T,step):
        vgmm.traj_gmm[i].plotCovs(ax)

    plt.suptitle("Trajectory - {}".format(Name))
    plt.tight_layout()
    plt.savefig("imgs/{}-Traj.pdf".format(Name))
    plt.show() 
    
def recordVideo(target,vgmm,Name):
    T=len(vgmm.traj_gmm)
    idx=0
    os.chdir(path+'/videos/')
    for t in np.arange(0,T,1):
        fig, (ax) = plt.subplots(1, 1,figsize=(10,5),num=0)
        target.plot(ax)
        vgmm.traj_gmm[t].plotCovs(ax)
        ax.set_aspect('equal')
        plt.savefig("GMM-{0:02d}.png".format(idx))
        plt.clf()
        idx=idx+1
    command=['/usr/local/bin/ffmpeg', '-framerate', '10', '-i', 'GMM-%02d.png', '-c:v','libx264','-pix_fmt', 'yuv420p','-y','{}.mp4'.format(Name)]
    subprocess.call(command)
    idx=0
    for t in np.arange(0,T,1):
        os.remove('GMM-{0:02d}.png'.format(idx))
        idx=idx+1
    
def plotHistoKL(target,listvgmm,listLabels,Name,num=0):
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    step=0
    #plt.yscale("log")
    os.chdir(path)
    for i in range(0,len(listvgmm)):
        vgmm=listvgmm[i]
        label=listLabels[i]
        T=len(listvgmm[i].traj_gmm)
        histo_kl=np.zeros([T,])
        for t in range(0,T):
            gmm=vgmm.traj_gmm[t]
            histo_kl[t]=gmm.KL(target,nbMC=100)
        plt.plot(range(0,T),histo_kl,label=label)
    
    if i>0:
        fig.legend(location="upper left")
    
    plt.xlabel("step")
    plt.ylabel("KL divergence")
    plt.suptitle("Gaussian-Mixture - {}".format(Name))
    plt.savefig("imgs/{}-KL.pdf".format(Name))
    plt.show() 

def computeResults(vgmm,target,Name,R=10,computePost=True,plotText=False,plot2D=False,plotKL=False,\
                   video=False,num=0):
    
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
        plotGMMcovs(target,vgmm,Name,num=num)
        num=num+1
        
    if plotKL:
        print("computing divergence...")
        plotHistoKL(target,[vgmm],["gmm"],Name,num=num)
        num=num+1
    
    if video:
        print("computing video...")
        recordVideo(target,vgmm,Name)

if __name__ == "__main__":
    ########## Choose here your desired test GMM-1, GMM-2, etc. ################

    TEST=["GMM-1"]
        
    if "GMM-1" in TEST:
        print("TEST GMM-1")
        d=2
        K=2
        step=0.1
        fixedWeights=False
        T=2.5
        invbeta=1
        Name="GMM2-2-s1"
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
        center=np.array([0,0])
        target.ComputeGrid2D(center,10,Npoints=Npoints)
       
        gmm=GMM(listw0,listMean0,GMM.cov2sqrt(listCov0))
        vgmm=VGMM(target,gmm,1/invbeta,fixedWeights=True)
        vgmm.propagate(step, T)
            
        plotGMMtraj(target,vgmm,"(mean=-10,0)",num=0,step=3)
        
    if "GMM-2" in TEST:
        print("TEST with 6 modes with isotropic cov with different weigths")
        d=2
        seed=1
        T=3
        step=1
        invbeta=1
        NModes=6
        np.random.seed(1)
        listw=np.random.uniform(1,3,NModes)
        listw=listw/listw.sum()
        listw=[0.2,0.1,0.2,0.2,0.05,0.15]
        Rtarget=5
        Name="GMM6-6-s1"
        
        # The target to approximate
        #target=generateGMM(NModes=NModes,listw=listw,R=Rtarget,scaleCov=1.5)
        target=generateGMM(NModes=NModes,listw=listw,R=Rtarget,scaleCov=1.5,randCov=True)

        # setting the initial Gaussians
        K=NModes
        Rmodel=10
        gmm=generateGMM(NModes=K,R=Rmodel)#,center=np.mean(target.means,axis=0))
            
        vgmmFix=VGMM(target,gmm,1/invbeta,fixedWeights=True)
        vgmmFix.propagate(step, T)
        #vgmmVar=VGMM(target,gmm,1/invbeta,fixedWeights=False)
        #vgmmVar.propagate(1, T)
        
        num=0
        computeResults(vgmmFix,target,Name=Name,R=max(Rtarget,Rmodel),\
                       plot2D=True,plotKL=False,video=True,num=num)
        num=num+1
            
        #computeResults(vgmmVar,target,Name=Name+'var',R=max(Rtarget,Rmodel),\
        #               plot2D=True,plotKL=False,video=True,num=num)
        #num=num+0
            
        #plotHistoKL(target,listvgmm=[vgmmFix,vgmmVar],\
        ##            listLabels=["fixed weights","variable weights"],Name=Name,num=num)
        #num=num+1
        
    if "GMM-3" in TEST:
        print("TEST with increasing number of samples")
        d=2
        seed=1
        T=3
        invbeta=1
        step=1
        NModes=6
        np.random.seed(1)
        listw=np.random.uniform(1,3,NModes)
        #listw=listw/listw.sum()
        listw=[0.1,0.1,0.4,0.2,0.05,0.15]
        Rtarget=5
        Name="GMM6-Test-Nbsamples"
        
        # The target to approximate
        target=generateGMM(NModes=NModes,listw=listw,R=Rtarget,scaleCov=1.5)
        #target=generateGMM(NModes=NModes,listw=listw,R=Rtarget,scaleCov=1.5,randCov=True)
        center=np.asarray(target.means).sum(axis=0)/2
        
        #setting the initial Gaussians
        listK=[NModes]#,2*NModes,5*NModes,10*NModes]
        listvgmm=[]
        listlabels=[]
        Rmodel=10
        for K in listK:
            gmm=generateGMMrandom(Nsamples=K,center=center,radius=Rmodel,edgeWidth=1)#,center=np.mean(target.means,axis=0))
            vgmm=VGMM(target,gmm,1/invbeta,fixedWeights=True)
            vgmm.propagate(step, T)
            listvgmm.append(vgmm)
            listlabels.append("{}-particles".format(K))
        
        
        
        num=0
        print("computing true posterior...")
        Npoints=100
        target.ComputeGrid2D(center,max(Rtarget,Rmodel),Npoints)
        for i in range(0,len(listvgmm)):
            computeResults(listvgmm[i],target,Name=listlabels[i],\
                        computePost=False,plot2D=True,plotKL=False,video=False,num=num)
            num=num+1

        num=num+1
        plotHistoKL(target,listvgmm,listlabels,Name=Name,num=num)
        
        # fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
        # gmm=generateGMMrandom(Nsamples=1000,center=center,radius=10,edgeWidth=0.1)
        # gmm.plotCovs(ax)
        # plt.show()

        
        
        
        
    