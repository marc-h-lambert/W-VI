###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Mathematical functions                                                          #
###################################################################################

import numpy.linalg as LA 
import numpy as np
import math
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt


class graphix: 
    # plot a 2D ellipsoid
    def plot_ellipsoid2d(ax,origin,Cov,col='r',zorder=1,label='',linestyle='dashed',linewidth=1):
        L=LA.cholesky(Cov)
        theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
        x = np.cos(theta)
        y = np.sin(theta)
        x,y=origin.reshape(2,1) + L.dot([x, y])
        ax.plot(x, y,linestyle=linestyle,color=col,zorder=zorder,label=label,linewidth=linewidth)
    
    # project a ND ellipsoid (mean-covariance) in plane (i,j)
    def projEllipsoid(theta,P,i,j):
        thetaproj=np.array([theta[i],theta[j]])
        Pproj=np.zeros((2,2))
        Pproj[0,0]=P[i,i]
        Pproj[0,1]=P[i,j]
        Pproj[1,0]=P[j,i]
        Pproj[1,1]=P[j,j]
        return thetaproj,Pproj
    
    def projEllipsoidOnVector(theta,P,v1,v2):
        x1=v1.T.dot(theta)
        x2=v2.T.dot(theta)
        thetaproj=np.array([x1,x2])
        v11=v1.T.dot(P).dot(v1)
        v22=v2.T.dot(P).dot(v2)
        v12=v1.T.dot(P).dot(v2)
        Pproj=np.array([[v11,v12], [v12,v22]])
        return thetaproj,Pproj

        

        
        
        