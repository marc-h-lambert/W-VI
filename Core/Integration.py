import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA

def rk4step(df,dt,X0,*args):
    """
    Runge Kutta algorithm of order 4 

    Parameters
    ----------
    df : the differential function
    dt : the integration step
    X0 : the state at t0
    
    Returns
    -------
    X : the state at t0 + dt
    """
    X=X0
    dX1=df(X,*args)
    X=X0+0.5*dt*dX1
    dX2=df(X,*args)
    X=X0+0.5*dt*dX2
    dX3=df(X,*args)
    X=X0+dt*dX3
    dX4=df(X,*args)
    X=X0+dt*(dX1+2.0*dX2+2.0*dX3+dX4)/6
    return X

def is_symmetric(x):
    return np.all(x==x.T)

def is_pos_def(x):
    return np.all(LA.eigvals(x) > 0)

def projectSDPold(A,debug,eps=1e-10):
    A=0.5*A+0.5*A.T
    d=A.shape[0]
    lmin=min(LA.eigvals(A))
    if lmin <eps:
        A=A+abs(lmin)*np.identity(d)
    return A

def projectSDP(A,debug,eps=1e-10):
    l,v=LA.eig(A)
    for i in range(0,l.shape[0]):
        if l[i]<0:
            if debug:
                print("negative eigen=",l[i])
            l[i]=-l[i]
        if l[i]==0:
            if debug:
                print("zero eigen=",l[i])
            l[i]=l[i]+eps
    A=v.dot(np.diag(l)).dot(v.T)
    A=0.5*A+0.5*A.T
    return A



def checkCov(P,s,debug=False):    
    if debug and not is_symmetric(P): 
        print("---> in ",s)
        print("Not symmetric matrix !!!!!!")
        print(P)
    if debug and not is_pos_def(P): 
        print("---> in ",s)
        print("Not def pos matrix  !!!!!!")
        print(P)
        print("eigs=",LA.eigvals(P))
    return projectSDP(P,debug)  

def checkState(X,d,s,debug=False):
    mu=X[0:d].reshape(-1,1)
    P=X[d:].reshape(d,d)
    P=checkCov(P,s,debug=debug)
    X=np.concatenate((mu, P.reshape(-1,1)), axis=0)
    return X

# Intregration on a vector (X , Cov X) of size d+d^2
# We force the Cov to be SDP along the integration process
def rk4stepSDP(df,dt,X0,d,*args):
    """
    Runge Kutta algorithm of order 4 

    Parameters
    ----------
    df : the differential function
    dt : the integration step
    X0 : the state at t0
    
    Returns
    -------
    X : the state at t0 + dt
    """
    X=X0
    X=checkState(X,d,"X0")
    dX1=df(X,d,*args)
    X=X0+0.5*dt*dX1
    X=checkState(X,d,"RK-step1")
    dX2=df(X,d,*args)
    X=X0+0.5*dt*dX2
    X=checkState(X,d,"RK-step2")
    dX3=df(X,d,*args)
    X=X0+dt*dX3
    X=checkState(X,d,"RK-step3")
    dX4=df(X,d,*args)
    X=X0+dt*(dX1+2.0*dX2+2.0*dX3+dX4)/6
    X=checkState(X,d,"RK-step4")
    return X

def integrateRK(df,T,dt,X0,*args):
    t=0
    X=X0
    d=X0.shape[0]
    traj=[]
    traj.append(np.concatenate(([t], X0.reshape(d,)), axis=0))
    while t < T:
        t=t+Decimal(dt)
        X=rk4step(df,dt,X,*args)
        traj.append(np.concatenate(([t], X.reshape(d,)), axis=0))
    return np.asarray(traj)

# Euler integration    
def IntegrateEuler(df,T,dt,X0):
    t=0
    X=X0
    d=X0.shape[0]
    traj=[]
    traj.append(np.concatenate(([t], X0.reshape(d,)), axis=0))
    while t < T: 
        t=t+Decimal(dt)
        X=X+df(X)*dt 
        traj.append(np.concatenate(([t], X.reshape(d,)), axis=0))
    return np.asarray(traj)

# Leap-Frog integration    
def stepLeapFrog(df,dt,X0,V0):
    V1=V0+df(X0)*dt/2
    X=X0+V1*dt
    V=V1+df(X)*dt/2
    return X,V

# Leap-Frog integration    
def integrateLeapFrog(df,T,dt,X0,V0):
    t=0
    X=X0
    V=V0
    d=X0.shape[0]
    traj=[]
    traj.append(np.concatenate(([t], X0.reshape(d,),V0.reshape(d,)), axis=0))
    while t < T: 
        t=t+Decimal(dt)
        X,V=stepLeapFrog(df,dt,X,V)
        traj.append(np.concatenate(([t], X.reshape(d,),V.reshape(d,)), axis=0))
    return np.asarray(traj)

# Euler-Marayama step
def StepSDE(df,D,dt,Q,X):
    d=X.shape[0]
    DeltaW=np.random.multivariate_normal(np.zeros(d,),Q*2*dt)
    DeltaW=DeltaW.reshape(d,1)
    return X+dt*df(X)+D.dot(DeltaW)

# Euler-Marayama integration    
def IntegrateSDE(df,D,T,dt,Q,X0):
    t=0
    X=X0
    d=X0.shape[0]
    traj=[]
    traj.append(np.concatenate(([t], X0.reshape(d,)), axis=0))
    while t < T: 
        t=t+Decimal(dt)
        X=StepSDE(df,D,dt,Q,X)
        traj.append(np.concatenate(([t], X.reshape(d,)), axis=0))
    #print('final X=',X)
    return np.asarray(traj)

# Trajectory convention format: matrix with row=state (nb lines = total time)
class Trajectory:
    # get the range and altitude of a given cartesian position 
    @staticmethod
    def range(pos,location0):
        r,lat,lon=Frame.cartToSphere(pos)
        rangeDist=location0.geoDistanceTo(geoLocation(lat,lon,0))
        return rangeDist
    
    @staticmethod
    def trajectoryFormat():
        return 'trajectory = [t, lat, lon, alt, rangeDist, state (GCC)]'
    
    @staticmethod
    def alt(pos):
        r,lat,lon=Frame.cartToSphere(pos)
        return r-Earth.EarthRadius

    # convert a SpaceVehicle trajectory [t, lat, lon, alt, rangeDist, state]
    # to a format t,x,y,z
    @staticmethod
    def trajPos(trajectory):
        time=trajectory[:,0:1] # prefer [:,0:1] to [:,0] to retrieve a 2D array !!
        xyz=trajectory[:,5:8]
        return np.hstack((time,xyz))

    # convert a SpaceVehicle trajectory [t, lat, lon, alt, rangeDist, state]
    # to a format t,x,y,z,vx,vy,vz
    @staticmethod
    def trajPosVel(trajectory):
        time=trajectory[:,0:1] # prefer [:,0:1] to [:,0] to retrieve a 2D array !!
        state=trajectory[:,5:11]
        return np.hstack((time,state))
    
    # traj format = t,x,y,z 
    @staticmethod
    def rangeAlt(traj,location0):
        nbPoints=traj.shape[0]
        rangeAlt=np.zeros([nbPoints,3])
        for t in np.arange(0,nbPoints):
            r,lat,lon=Frame.cartToSphere(traj[t,1:4])
            alt=r-Earth.EarthRadius
            rangeDist=location0.geoDistanceTo(geoLocation(lat,lon,0))
            rangeAlt[t,0]=traj[t,0]
            rangeAlt[t,1]=rangeDist
            rangeAlt[t,2]=alt
        return rangeAlt
    
    # find a tome associated to a desired altitude 
    # from the end point to the start point
    @staticmethod
    def timeAtAlt(trajectory,alt):
        finalTime=trajectory[-1,0]
        t=finalTime
        for i in range(1,trajectory.shape[0]):
            if trajectory[-i,3]>alt:
                return trajectory[-i,0]
            
    @staticmethod
    def findAlt(trajectory,alt):
        for i in range(1,trajectory.shape[0]):
            if trajectory[-i,3]>alt:
                return trajectory.shape[0]-i
        return -1
            
    @staticmethod
    def findDate(trajectory,date,epsilon=0.01):
        for i in range(1,trajectory.shape[0]):
            if abs(trajectory[-i,0]-date)<epsilon:
                return trajectory.shape[0]-i
        return -1
    
