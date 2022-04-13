import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from Core.VariationalGP import VGP_JKO, VGP_bank, GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import special_ortho_group

def covariance(diag,rotate=True):
    vec=diag/LA.norm(diag)**2
    Cov_u=np.diag(vec)
    if rotate:
        Q = special_ortho_group.rvs(dim=diag.shape[0])
        Cov_u=np.transpose(Q).dot(Cov_u).dot(Q)
    return Cov_u
    

if __name__ == "__main__":
    ########## Choose here your desired test GP-1, GP-2, etc. ################
    TEST = ["GP-4"]
    
    if "GP-1" in TEST: # test with one Gaussian
        d=2
        # The target to approximate
        listMean=np.array([[2, 2], [6, 6]])#, [2, 1],[0, 1]]
        listCov=np.array([np.identity(2), np.identity(2)])#, 0.3*np.identity(2),0.3*np.identity(2)]
        listw=np.array([0.5,0.5])#,0.25,0.25]
        target=GaussianMixture(listw,listMean,listCov)
        
        # setting the initial Gaussians
        mean0=np.zeros([2,1])
        mean0[0]=0
        mean0[1]=0
        cov0=np.identity(2)
        beta=1
        
        # running the banck of filters
        dt=1
        T=100
        vgp=VGP_JKO(target,mean0,cov0,beta)
        print("vgp.mean before=",vgp.mean)
        vgp.propagate(dt,T)
        print("vgp.mean after=",vgp.mean)
        
        # plot results
        num=1
        fig, (ax) = plt.subplots(1, 1,figsize=(10,5),num=num)
        vgp.plot(ax, 50)
        center=listMean.sum(axis=0)/2
        radius=10
        Npoints=100
        target.ComputeGrid2D(center,radius,Npoints)
        target.plot(ax)
        plt.show()
        
    if "GP-2" in TEST: # test with N Gaussians
        d=2
        # The target to approximate
        listMean=np.array([[2, 2], [6, 6]])#, [2, 1],[0, 1]]
        listCov=np.array([np.identity(2), np.identity(2)])#, 0.3*np.identity(2),0.3*np.identity(2)]
        listw=np.array([0.5,0.5])#,0.25,0.25]
        target=GaussianMixture(listw,listMean,listCov)
        # usefull for drawing
        print("computing true posterior...")
        center=listMean.sum(axis=0)/2
        radius=10
        Npoints=100
        target.ComputeGrid2D(center,radius,Npoints)
            
        # setting the initial Gaussians
        listMean0=np.array([[0, 0], [1, 0], [-1,0], [0,1], [0,-1]])*10
        listCov0=np.array([np.identity(2), np.identity(2), np.identity(2), \
                       np.identity(2), np.identity(2)])
        beta=1
        
        # running the bank of filters
        dt=1
        T=10
        
        vgpBank=VGP_bank(target,listMean0,listCov0,beta)
        vgpBank.runFilters(dt,T)
        
        # plot results
        num=0
        for t in range(0,int(T/dt)):
            fig, (ax) = plt.subplots(1, 1,figsize=(10,5),num=num)
            target.plot(ax)
            for i in range(0,vgpBank.Nfilters):
                vgpBank.listVGP[i].plot(ax, t)
            plt.savefig("videos/GP2-{0:02d}.png".format(t))
            if t==0:
                plt.title("Initial configuration")
                plt.show()
            if t==int(T/dt)-1:
                plt.title("Final configuration after {} steps".format(int(T/dt)))
                plt.show()
            plt.clf()
    
    if "GP-3" in TEST: # test with N Gaussians
        d=2
        seed=1
        # The target to approximate
        # setting the initial Gaussians
        NModes=6
        R=5
        delta=2*math.pi/NModes
        listMean=[]
        listCov=[]
        listw=[]
        theta=0
        for i in range(0,NModes):
            listMean.append([R*math.cos(theta),R*math.sin(theta)])
            cov=np.identity(2)*1.5
            #cov[0,1]=np.random.uniform(0,1)
            #cov[1,0]=cov[0,1]
            #cov[0,0]=np.random.uniform(1,1.5)
            listCov.append(cov)
            listw.append(1/NModes)
            theta=theta+delta
        #listw=np.array([0.2,0.2,0.2,0.2,0.2])
        target=GaussianMixture(np.asarray(listw),np.asarray(listMean),np.asarray(listCov))
        # usefull for drawing
        print("computing true posterior...")
        center=np.asarray(listMean).sum(axis=0)/2
        radius=10
        Npoints=100
        target.ComputeGrid2D(center,radius,Npoints)
            
        # setting the initial Gaussians
        Nfilters=10
        R=10
        delta=2*math.pi/Nfilters
        listMean0=[]
        listCov0=[]
        theta=0
        for i in range(0,Nfilters):
            listMean0.append([R*math.cos(theta),R*math.sin(theta)])
            cov=np.identity(2)
            listCov0.append(cov)
            theta=theta+delta
        beta=1
        
        # running the bank of filters
        dt=1
        T=10
        
        vgpBank=VGP_bank(target,np.asarray(listMean0),np.asarray(listCov0),beta)
        vgpBank.runFilters(dt,T)
        
        # plot results
        num=0
        for t in range(0,int(T/dt)):
            fig, (ax) = plt.subplots(1, 1,figsize=(10,5),num=num)
            target.plot(ax)
            for i in range(0,vgpBank.Nfilters):
                vgpBank.listVGP[i].plot(ax, t)
            plt.savefig("videos/GP3-{0:02d}.png".format(t))
            if t==0:
                plt.title("Initial configuration")
                plt.show()
            if t==int(T/dt)-1:
                plt.title("Final configuration after {} steps".format(int(T/dt)))
                plt.show()
            plt.clf()
        #os.chdir("videos")
        #subprocess.call('ffmpeg')subprocess.call('ffmpeg')#['ffmpeg', '-framerate', '8', '-i', 'GP2-%02d.pdf', '-r', '30', '-pix_fmt', 'yuv420p','GP2.mp4'])
        
    if "GP-4" in TEST: # test with N Gaussians
        d=2
        
        seed=1
        np.random.seed(seed)
        # The target to approximate
        # setting the initial Gaussians
        NModes=6
        R=4
        delta=2*math.pi/NModes
        listMean=[]
        listCov=[]
        listw=[]
        theta=0
        XP=6
        covscale=1
        for i in range(0,NModes):
            listMean.append([R*math.cos(theta),R*math.sin(theta)])
            cov=covariance(np.asarray([1,3]),rotate=True)*10
            #cov=np.identity(2)*covscale

            #cov[0,1]=np.random.uniform(0,1)
            #cov[1,0]=cov[0,1]
            #cov[0,0]=np.random.uniform(1,1.5)
            listCov.append(cov)
            listw.append(1/NModes)
            #listw[0]=0.8
            theta=theta+delta
        
        listw=np.asarray(listw)
        listw=listw/listw.sum()
        #listw=np.array([0.2,0.2,0.2,0.2,0.2])
        target=GaussianMixture(listw,np.asarray(listMean),np.asarray(listCov))
        # usefull for drawing
        print("computing true posterior...")
        center=np.asarray(listMean).sum(axis=0)/2
        radius=10
        Npoints=100
        target.ComputeGrid2D(center,radius,Npoints)
            
        # setting the initial Gaussians
        Nfilters=10
        R=10
        delta=2*math.pi/Nfilters
        listMean0=[]
        listCov0=[]
        theta=0
        for i in range(0,Nfilters):
            listMean0.append([R*math.cos(theta),R*math.sin(theta)])
            cov=np.identity(2)
            listCov0.append(cov)
            theta=theta+delta
        beta=1
        
        # running the bank of filters
        dt=1
        T=10
        
        vgpBank=VGP_bank(target,np.asarray(listMean0),np.asarray(listCov0),beta)
        vgpBank.runFilters(dt,T)
        
        # plot results
        num=0
        fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5),num=num)
        target.plot(ax1)
        for i in range(0,vgpBank.Nfilters):
            vgpBank.listVGP[i].plot(ax1, 0)
        target.plot(ax2)
        for i in range(0,vgpBank.Nfilters):
            vgpBank.listVGP[i].plot(ax2, T-1)
        plt.suptitle("Gaussian-Mixture: equal weigts, non isotropic covariances")
        plt.savefig("images/GP4-{}.pdf".format(XP))
        plt.show()
        
    if "GP-5" in TEST: # test with N Gaussians
        d=2
        
        seed=1
        np.random.seed(seed)
        # The target to approximate
        # setting the initial Gaussians
        NModes=6
        R=5
        delta=2*math.pi/NModes
        listMean=[]
        listCov=[]
        listw=[]
        theta=0
        XP=9
        covscale=1
        for i in range(0,NModes):
            listMean.append([R*math.cos(theta),R*math.sin(theta)])
            #cov=covariance(np.asarray([1,3]),rotate=True)*10
            cov=np.identity(2)*covscale

            #cov[0,1]=np.random.uniform(0,1)
            #cov[1,0]=cov[0,1]
            #cov[0,0]=np.random.uniform(1,1.5)
            listCov.append(cov)
            listw.append(1/NModes)
            #listw[0]=0.8
            theta=theta+delta
        
        listw=np.asarray(listw)
        listw=listw/listw.sum()
        #listw=np.array([0.2,0.2,0.2,0.2,0.2])
        target=GaussianMixture(listw,np.asarray(listMean),np.asarray(listCov))
        # usefull for drawing
        print("computing true posterior...")
        center=np.asarray(listMean).sum(axis=0)/2
        radius=10
        Npoints=100
        target.ComputeGrid2D(center,radius,Npoints)
            
        # setting the initial Gaussians
        Nfilters=10
        R=2
        delta=2*math.pi/Nfilters
        listMean0=[]
        listCov0=[]
        theta=0
        for i in range(0,Nfilters):
            listMean0.append([R*math.cos(theta),R*math.sin(theta)])
            cov=np.identity(2)*100
            listCov0.append(cov)
            theta=theta+delta
        beta=1
        
        # running the bank of filters
        dt=1
        T=10
        
        vgpBank=VGP_bank(target,np.asarray(listMean0),np.asarray(listCov0),beta)
        vgpBank.runFilters(dt,T)
        
        # plot results
        num=0
        fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5),num=num)
        target.plot(ax1)
        for i in range(0,vgpBank.Nfilters):
            vgpBank.listVGP[i].plot(ax1, 0)
        target.plot(ax2)
        for i in range(0,vgpBank.Nfilters):
            vgpBank.listVGP[i].plot(ax2, T-1)
        plt.suptitle("Gaussian-Mixture: equal weigts, non isotropic covariances")
        plt.savefig("images/GP4-{}.pdf".format(XP))
        plt.show()

    