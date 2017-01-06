import numpy as np, pylab as pl, pyfits as py
import cPickle

alpha,beta,gamma = 1.25,0.3,-7.

logsigma = np.random.randn(100)*0.5 + 2.35
mu = np.random.randn(100)*1. + 20.

re = alpha*logsigma + beta*mu + gamma + np.random.randn(100)*0.2
