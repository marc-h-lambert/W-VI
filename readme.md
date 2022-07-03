# Variational inference via Wasserstein gradient flows (W-VI)

## Object

This is the companion code for the paper \[[1][4]\]. Please cite this paper if you use this code.  

## Installation
The code is available in python using the standard library. 

## python files
The source of the algorithms which implement the Wasserstein variational inference is available in python [here][0]. Boths case are managed: approximation of a target distribution with a Gaussian or with a mixture of Gaussian. In each case the target can be a mixture of Gaussians or a logistic target. [VGA_main][1] is the main program to test the variational Gaussian approximation (model with only one Gaussian) whereas [GMM_main][2] is the corresponding test program for mixture of Gaussian model. Both can be run to reproduce some results of the paper run. Notice that only the tests appearing in the string list TEST are run, modify this list in the code to add your desired tets. 

[0]: ./Core
[1]: ./VGA_main.py
[2]: ./GMM_main.py
[4]: ...

\[1\]: ["Variational inference via Wasserstein gradient flows , M. Lambert, S. Chewi, S. Bonnabel, F. Bach, P. Rigollet."][4] 
