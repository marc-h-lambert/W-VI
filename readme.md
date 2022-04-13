# JKO scheme on the space of Gaussians distribution (Bures-JKO)

## Object

This is the companion code for the NIPS paper project.  

## Installation
The code is available in python using the standard library. Run TestVGP.py. To change the running test, edit the file 
and put another string in the array TEST=["MyTest"].

## python files
The source of the library "Core" which implement the assesed algorithms is available in python [here][0] and 
structured as follows:
- VariationalGP.py: contain the model for the target (class GaussianMixture), the Gaussian process (class VGP_JKO) and the bank of filters for multi-gaussian sampling (class VGP_bank). The GaussianMixture class include tools to compute and show the true distribution.
- Integration.py: a set of tools to integrate an ODE. Only the class rk4step (RuneKutta integrator of order 4) is used here.
- graphix.py: tools to draw ellipses


[0]: ./Core


