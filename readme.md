# Variational inference via Wasserstein gradient flows (W-VI)
(TO UPDATE)
## Object

This is the companion code for the NIPS paper project.  

## Installation
The code is available in python using the standard library. Run TestVGP.py. To change the running test, edit the file 
and put another string in the array TEST=["MyTest"].

## python files
The source of the library "Core" which implement the assesed algorithms is available in python [here][0] and 
structured as follows:
- VariationalGP.py: contain the model for the Gaussian process governed by Sarkka's ODE. and the bank of filters for multi-gaussian sampling (class VGP_bank). 
- VariationalGMM.py: contain the model for the GaussianMixture class including tools to compute and show the true distribution. Contains also the class VGMM whihc implenet the algorithm to solve the Wasserstein gradient flows on GMM.
- Integration.py: a set of tools to integrate an ODE. Only the class rk4step (RuneKutta integrator of order 4) is used here.
- graphix.py: tools to draw ellipses


[0]: ./Core


