from ast import MatMult
from locale import normalize
from operator import matmul
import sunau
from turtle import shape
import numpy as np

p = print


## q1 ###

N = 50
theta = np.array([1.0, 1.0, 1.0])
rho = 0.0
mu_x = np.array([0.0, 0.0])
vol_x = np.array([2.0, 2.0])
cov_x = np.outer(vol_x, vol_x) * np.array([[1.0, rho], [rho, 1.0]])

X = np.random.multivariate_normal(mu_x, cov_x, size=N)
X = np.column_stack((np.ones(50), X))

eps = np.random.normal(size=50)

Y = X @ theta + eps

''' @ == mmult '''

### OLS- reg ###
# manual reg: 

rho_est = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), (np.matmul(np.transpose(X),Y)))

#direct reg 
theta, resid, rank, s = np.linalg.lstsq(X, Y, rcond=-1) 

print(("""OLS parameter estimates: \n
alpha = {:.10f}
beta_x1 = {:.10f}
beta_x2 = {:.10f}""").format(*theta)) ## Format like theta!! 

''' Signular values => Sqrt of non-zero eigenvalues (lambda)'''


