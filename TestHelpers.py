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
from Core.Utils import Expect
from Core.LangevinTarget import logisticpdf, logisticpdfHD
from Core.Laplace import LaplaceLogReg
from plotHelpers import generateGMM, generateGMMrandom, generateGMMsquare,\
    generateCovariance, plotGMMcovs, plotGMMtraj, \
        recordVideo, plotHistoKL, plotHistoKLgaussian, KLnormalVslogpdf, \
            computeResults, plotHistoRightKL
from scipy.stats import multivariate_normal
from scipy.stats import special_ortho_group

def TestSingleGaussianSampling(target,Name,K,T=10,center=[],Nfigs=4,Rmodel=12,num=0,\
                               stepRK=1,stepPlot=1,x1=-15,x2=15,y1=-15,y2=15):
    d=2
    seed=2
    T=T
    stepRK=stepRK
    invbeta=1
    stepPlot=int(1/stepRK)

    Rmodel=Rmodel
    adaptStep=False 
    fixedWeights=True
    
    np.random.seed(seed)
    if len(center)==0:
        center=target.mean()
    print(center)
    gmm=generateGMMrandom(Nsamples=K,center=center,radius=Rmodel,\
                              scaleCov=1,edgeWidth=1)
    vgmm=VGMM(target,gmm,1/invbeta,fixedWeights=fixedWeights)
    print("Propagate")
    vgmm.propagate(stepRK, T)
    
    plotGMMcovs(target,vgmm,Name,center=center,num=0,Ncols=Nfigs,R=Rmodel,\
                adaptStep=False,stepSize=stepPlot,\
                x1=x1,x2=x2,y1=y1,y2=y2)
        
def TestGaussianSampling(target,Name,listK=[6],Rmodel=12,num=0):
    d=2
    seed=1
    T=5
    stepRK=1
    invbeta=1
    stepPlot=int(1/stepRK)
    #setting the initial Gaussians
    listK=listK
    listvgmm=[]
    listlabels=[]
    Rmodel=Rmodel
    adaptStep=False #to adapt step plots in function of GMM modes
    fixedWeights=True
    for K in listK:
        np.random.seed(seed)
        center=target.mean()
        gmm=generateGMMrandom(Nsamples=K,center=center,radius=Rmodel,\
                              scaleCov=1,edgeWidth=1)
        vgmm=VGMM(target,gmm,1/invbeta,fixedWeights=fixedWeights)
        label="{}-{}-particles".format(Name,K)
        print("Propagate-",label)
        vgmm.propagate(stepRK, T)
        listvgmm.append(vgmm)
        listlabels.append(label)
    
    # print("computing true posterior...")
    # Npoints=100
    # Rdraw=Rmodel#max(Rtarget,Rmodel)
    # target.ComputeGrid2D(max(Rtarget,Rmodel),Npoints)
    # for i in range(0,len(listvgmm)):
    #     computeResults(listvgmm[i],target,Name=listlabels[i],R=Rdraw,\
    #                 computePost=False,plot2D=True,plotKL=False,video=False,\
    #                     num=num,adaptStep=adaptStep,stepSize=stepPlot)
    #     num=num+1

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
                listnormalSamples=listnormalSamples,adaptStep=adaptStep,stepSize=stepPlot)
    
    #computeResults(listvgmm[-1],target,Name=Name+"-"+listlabels[-1],R=Rdraw,\
    #               plot2D=False,plotKL=False,video=True,num=num,ContourLines=False,\
    #                   adaptStep=adaptStep,stepSize=stepPlot)
    #num=num+T
    #computeResults(listvgmm[-1],target,Name=Name+"-"+listlabels[-1]+"-density",R=Rdraw,\
    #               plot2D=False,plotKL=False,video=True,num=num,ContourLines=True)

def TestCKF(N,s,d,num):
    target=logisticpdfHD(s=s,N=N,d=d)
    Y,X=target.Obs.datas
    Y=Y.reshape(N,)    
    lap=LaplaceLogReg(np.zeros([d,1]),np.identity(d))
    lap.fit(X,Y)
    text="Expectation of logistic distribution under Laplace Gaussian "
    Expect.PlotCKFvsMC(target.logpdf,lap.mode,LA.cholesky(lap.Cov),num,text)
    plt.savefig("CKF1.pdf")
    num=num+1
    
    # The initial Gaussian 
    K=1
    listw0=np.ones([K,1])
    listw0=listw0/listw0.sum(axis=1)
    mean0=np.zeros([d, ])
    cov0=np.identity(d)*100
    listMean0=[mean0]
    listCov0=[]
    for i in range(0,K):
        listCov0.append(cov0)
    gmm=GMM(listw0,listMean0,GMM.cov2sqrt(listCov0))
    
    # run Wasserstein VI
    vgmm=VGMM(target,gmm,1,fixedWeights=True)
    vgmm.propagate(0.1, 10)
    gmmStar=vgmm.traj_gmm[-1]
    text="Expectation of logistic distribution under VI Gaussian "
    Expect.PlotCKFvsMC(target.logpdf,gmmStar.means[0],gmmStar.rootcovs[0],num,text)
    num=num+1
    plt.savefig("CKF2.pdf")
    
def TestLaplace2D(N,s,radius=8,scaleCov=1,num=0,Npoints=100):
    d=2
    step=0.1
    stepPlot=int(1/step)
    fixedWeights=False
    T=30
    invbeta=1
    TitleText="Logistic regression - step size {}".format(step)
    FileName="GaussianVI-LogReg-d{}-N{}-s{}".format(d,N,int(s))
    
    # The target to approximate
    target=logisticpdf(s=s,N=N,d=d)
    print("computing true posterior...")
    Npoints=Npoints
    center=np.array([0,0])
    target.ComputeGrid2D(10,Npoints=Npoints,center=center)
    center=target.mean()
    target.ComputeGrid2D(radius,Npoints=Npoints,center=center)
    x1=center[0]-radius
    x2=center[0]+radius
    y1=center[1]-radius
    y2=center[1]+radius
    
    # The initial Gaussian 
    K=1
    listw0=np.ones([K,1])
    listw0=listw0/listw0.sum(axis=1)
    mean0=np.array([-10, 0])
    cov0=np.identity(2)*scaleCov
    listMean0=[mean0]
    listCov0=[]
    for i in range(0,K):
        listCov0.append(cov0)
    gmm=GMM(listw0,listMean0,GMM.cov2sqrt(listCov0))
    
    # run Wasserstein VI
    vgmm=VGMM(target,gmm,1/invbeta,fixedWeights=True)
    vgmm.propagate(step, T)
    gmmStar=vgmm.traj_gmm[-1]
        
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
    plotGMMtraj(target,vgmm,TitleText,FileName,num=num,step=1,coefStep=2,\
                x1=x1,x2=x2,y1=y1,y2=y2)
    #ax.set_aspect("equal")
    num=num+1
    
    # plot 2D cov lap vs VI
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    num=num+1
    gmmStar.plotCovs(ax,label="Wasserstein-VI")
    target.plot(ax)
    target.plotEllipsoid(ax,col='k',label="True moments")
    print("pdf at lap mode=",target.pdf(lap.mode))
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
    
    # plot 3D VI
    dpi=100
    angle=-20#85
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    ax = plt.axes(projection='3d',rasterized=True)
    ax._axis3don = False
    ax.view_init(30, angle)
    plt.xticks([], [])
    plt.yticks([], [])
    target.plot3D(ax)
    gmmStar.plot3D(ax,target.xv,target.yv)
    plt.title("Wasserstein-VI")
    plt.tight_layout()
    plt.savefig("imgs/{}-3DplotVI.pdf".format(FileName),dpi=dpi)
    num=num+1
    
    # plot 3D Laplace
    fig, ax = plt.subplots(1, 1,figsize=(3,3),num=num)
    ax = plt.axes(projection='3d',rasterized=True)
    ax._axis3don = False
    ax.view_init(30, angle)
    plt.xticks([], [])
    plt.yticks([], [])
    target.plot3D(ax)
    lap.plot3D(ax,target.xv,target.yv)
    plt.title("Laplace")
    plt.tight_layout()
    plt.savefig("imgs/{}-3DplotLap.pdf".format(FileName),dpi=dpi)
    num=num+1
        
    # plot left KL
    #nbMC=1000
    #print("target.Z=",target.Z)
    ##normalSamples=np.random.multivariate_normal(np.zeros(d,), np.identity(d),size=(nbMC,))
    #weightSamples=np.random.choice(np.arange(0,gmm.K), size=(nbMC,), p=gmm.weights.reshape(-1,))
    #
    #left_kl_lap=lapgmm.KLseed(target,weightSamples,normalSamples)
    #left_kl_vi=gmmStar.KLseed(target,weightSamples,normalSamples)
    #print("Left KL Laplace=",left_kl_lap)
    #print("Left KL VI=",left_kl_vi)
    #plotHistoKL(target,[vgmm],["VI"],Name=FileName,num=num,listweightSamples=[weightSamples],\
    #            listnormalSamples=[normalSamples],adaptStep=False,stepSize=stepPlot,kl_laplace=left_kl_lap,unnormalized=False)
    
    nbMC=1000
    normalSamples=np.random.multivariate_normal(np.zeros(d,), np.identity(d),size=(nbMC,))
    R=LA.cholesky(lap.Cov)
    left_kl_lapv2=KLnormalVslogpdf(lap.mode,R,target.logpdf)
    print("Left KL Laplace v2=",left_kl_lapv2)
    plotHistoKLgaussian(target,[vgmm],["VI"],Name=FileName,num=num,\
              stepSize=stepPlot,kl_laplace=left_kl_lapv2,unnormalized=False)
    num=num+1
        
    # plot right KL
    samples=[]
    for i in range(0,nbMC):
        samples.append(target.random())
    samples=np.array(samples)
    lapgmm=GMM(listw0,[lap.mode],GMM.cov2sqrt([lap.Cov]))
    right_kl_lap=lapgmm.RightKLseed(target,samples)
    right_kl_vi=gmmStar.RightKLseed(target,samples)
    print("Right KL Laplace=",right_kl_lap)
    print("Right KL VI=",right_kl_vi)
    plotHistoRightKL(target,[vgmm],["Bures-Wasserstein-VI"],Name=FileName,num=num,\
                listSamples=[samples],adaptStep=False,stepSize=stepPlot,kl_laplace=right_kl_lap,unnormalized=False)

def TestHD(N,s,d,scaleCovVI=1,scaleCovLap=-1,num=0,Z=1,scale=1):
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
    target=logisticpdfHD(s=sep,N=N,d=d,Z=Z)
    
    # The initial Gaussian 
    K=1
    listw0=np.ones([K,1])
    listw0=listw0/listw0.sum(axis=1)
    mean0=np.zeros([d, ])
    mean1=np.ones([d, ])
    #mean0[0]=1
    cov0=np.identity(d)*scaleCovVI
    listMean0=[mean0]
    listCov0=[]
    for i in range(0,K):
        listCov0.append(cov0)
    gmm=GMM(listw0,listMean0,GMM.cov2sqrt(listCov0))
    gmm2=GMM(listw0,[mean1],GMM.cov2sqrt(listCov0))
    
    nbMC=1000
    print("target.Z=",target.Z)
    normalSamples=np.random.multivariate_normal(np.zeros(d,), np.identity(d),size=(nbMC,))
    weightSamples=np.random.choice(np.arange(0,gmm.K), size=(nbMC,), p=gmm.weights.reshape(-1,))
    
    kl=gmm.KLseed(target,weightSamples,normalSamples)
    kl2=gmm2.KLseed(target,weightSamples,normalSamples)
    print("Left KL Mean0=",kl)
    print("Left KL Mean1=",kl2)
    
    # run Wasserstein VI
    vgmm=VGMM(target,gmm,1/invbeta,fixedWeights=True)
    #vgmm.propagate(step, T)
    #gmmStar=vgmm.traj_gmm[-1]
        
    # run Laplace (non regularized)
    Y,X=target.Obs.datas
    Y=Y.reshape(N,)   
    if scaleCovLap>0:
        lap=LaplaceLogReg(mean0,scaleCovLap*np.identity(d))
        print("here")
    else:
        lap=LaplaceLogReg(mean0)
    lap.fit(X,Y)
    
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
    lapgmm=GMM(listw0,[lap.mode],GMM.cov2sqrt([lap.Cov]))
    if not (d==100 and s==10):
        left_kl_lap=lapgmm.KLseed(target,weightSamples,normalSamples)
    else:
        left_kl_lap=None
    #left_kl_vi=gmmStar.KLseed(target,weightSamples,normalSamples)
    print("Left KL Laplace=",left_kl_lap)
    #print("Left KL VI=",left_kl_vi)
    plotHistoKL(target,[vgmm],["VI"],Name=FileName,num=num,listweightSamples=[weightSamples],\
                listnormalSamples=[normalSamples],adaptStep=False,stepSize=stepPlot,\
                kl_laplace=left_kl_lap)
    num=num+1
     
    R=LA.cholesky(lap.Cov)
    if not (d==100 and s==10):
        left_kl_lapv2=KLnormalVslogpdf(lap.mode,R,target.logpdf)
    else:
        left_kl_lapv2=None
    print("Left KL Laplace v2=",left_kl_lapv2)
    plotHistoKLgaussian(target,[vgmm],["VI"],Name=FileName,num=num,\
              stepSize=stepPlot,kl_laplace=left_kl_lapv2,text=" d={0}, N={1}, s={2:.2g}".format(d,N,s))
    num=num+1
        
def BiModalTarget():
    listMean=np.array([[5, 3.3], [5, -3.3]])
    listCov=np.array([np.diag([3,1]), np.diag([1,3])])
    listw=np.array([0.5,0.5])
    target=GMM(listw,np.asarray(listMean),GMM.cov2sqrt(listCov))
    return target


