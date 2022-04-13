# The Recursive variational gaussian approximation (R-VGA)

## Object

This is the companion code for the paper \[[1][4]\]. Please cite this paper if you use this code.  

## Installation
The code is available in python using the standard library. We depends on the "scipy" library for optimization, we use the scipy powell method to compute the implicit scheme and the scipy l-bfgs method to compute the maximum posterior for Laplace approximation. If not available on your distribution, you may install scipy following https://www.scipy.org/install.html.

## python files
The source of the library "Kalman Machine" which implement the assesed algorithms is available in python [here][0]. To test this library on an exemple case, run [TestKalmanMachine][1]. To reproduce the results of the paper run [PaperResults][2]. 

## Tutorial in jupyter notebook
To go further on this subject, a set of tutorial will be available [here][3] (not yet published). 

[0]: ./KalmanMachine
[1]: ./TestKalmanMachine.py
[2]: ./PaperResults.py
[3]: ./Tutorial/README.md
[4]: https://hal.inria.fr/hal-03086627 

\[1\]: ["Recursive variational gaussian approximation (R-VGA), Marc Lambert, Silvere Bonnabel and Francis Bach".][4] 
