import numpy as np, pylab as pl, pyfits as py

n1,n2 = np.load('/data/ljo31/Lens/LensParams/structurecat_srcs_2src.npy').T
ii=np.where(n1>0)
n1,n2=n1[ii],n2[ii]
for ii in range(n1.size):
    a,b=n1[ii],n2[ii]
    n1[ii],n2[ii] = np.min((a,b)), np.max((a,b))
n = np.load('/data/ljo31/Lens/LensParams/structurecat_srcs_1src.npy')
nl1,nl2 = np.load('/data/ljo31/Lens/LensParams/structurecat_lensgals.npy').T
for ii in range(nl1.size):
    a,b=nl1[ii],nl2[ii]
    nl1[ii],nl2[ii] = np.min((a,b)), np.max((a,b))
bins = np.linspace(0.1,12,24)

pl.figure()
pl.hist(n1,bins,label='nuggets',alpha=0.5,normed=1)
pl.hist(nl1,bins,label='lenses',alpha=0.5,normed=1)
pl.title('disky components')
pl.xlabel('n')
pl.legend(loc='upper right')
pl.figure()
pl.hist(n2,bins,label='nuggets',alpha=0.5,normed=1)
pl.hist(nl2,bins,label='lenses',alpha=0.5,normed=1)
pl.title('bulgey components')
pl.legend(loc='upper right')
pl.xlabel('n')
pl.figure()
pl.hist(n,bins,label='nuggets',alpha=0.5,normed=1)
pl.title('single components')
pl.xlabel('n')
pl.legend(loc='upper right')
