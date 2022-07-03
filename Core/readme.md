
# The Kalman machine library (v3)

## Object

This is the third version of the Kalman machine library dedicated to Wasersstein gradient flow. This version only deal with the continuous case (i.e. with ODE) in low dimension. The discrete case and factor analysis variant are available on the repository [L-RVGA][9], notice the the code in [RVGA][10] is now deprecated. 

## Installation
The code is available in python using the standard library. 

## python files
The class  [VariationalGP][0] implement the variational Gaussian approximation. It integrate the W-VI ODE with the sigma points method throught the API propagate. The integration tools are available in [Integration][7]. The class [VariationalGMM][1] implement VI for a mixture of Gaussian model, it integrate a set of ODE (two ODE by Gaussian) also with the sigma points method. The quadrature rules based on sigma points methods are described in [Utils][6]. The class [GMM][2] contain a mixture of Gaussian structure and can be used to describe a target or a GMM model. All the targets must satisfy the API described in [LangevinTarget][3] which contain also a description for the logistic target build upon a synthetic dataset [SyntheticDataset][4]. The class [Laplace][5] contain the Laplace method used to compare our algorithm. Finally the class [Graphix][8] is used mainly to draw ellipsoids.       

[0]: ./VariationalGP.py
[1]: ./VariationalGMM.py
[2]: ./GMM.py
[3]: ./LangevinTarget.py
[4]: ./SyntheticDataset.py
[5]: ./Laplace.py
[6]: ./Utils.py
[7]: ./Integration.py
[8]: ./Graphix.py
[9]: https://github.com/marc-h-lambert/L-RVGA/tree/main/KalmanMachine
[10]: https://github.com/marc-h-lambert/RVGA/tree/main/KalmanMachine

