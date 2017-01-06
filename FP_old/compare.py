import numpy as np, pylab as pl, pyfits as py, cPickle

sigmaE,muE,reE,dsE,dmE,drE = np.load('/data/ljo31b/EELs/esi/kinematics/FP_EELs_mu.npy').T

sigmaM,muM,reM,dsM,dmM,drM = np.load('/data/ljo31b/EELs/esi/kinematics/FP_MACS_mu.npy').T

# FP
pl.figure()
pl.scatter(1.24*sigmaE+0.3*muE,reE,s=30,color='SteelBlue')
pl.scatter(1.24*sigmaM+0.3*muM,reM,s=30,color='Crimson')
pl.title('FP')

# FJR
pl.figure()
pl.scatter(sigmaE, muE+5.*reE,color='SteelBlue')
pl.scatter(sigmaM,muM+5.*reM,s=30,color='Crimson')
pl.title('FJR')

# KR
pl.figure()
pl.scatter(reE,muE,color='SteelBlue')
pl.scatter(reM,muM,s=30,color='Crimson')
pl.title('KR')

# they seem to be vaguely consistent, but it's possible the EELs are shallower. If they have overmassive haloes, they have high M/L ratios, 
