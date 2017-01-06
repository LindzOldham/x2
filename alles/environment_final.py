import sqlutil, numpy as np, pylab as pl, pyfits as py
from astLib import astCalc

def clip(arr,nsig=4.5):
    a = np.sort(arr.ravel())
    a = a[a.size*0.001:a.size*0.999]
    while 1:
        m,s,l = a.mean(),a.std(),a.size
        a = a[abs(a-m)<nsig*s]
        if a.size==l:
            return m,s

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]

Nwsamples = np.load('/data/ljo31b/EELs/environments/Nwsamplesdict.npy')[()]
Nw = np.load('/data/ljo31b/EELs/environments/Nwdict.npy')[()]

allsamples = []
allseels = []
for name in sz.keys():
    if name == 'J1248':
        continue
    nwsamp = np.array(Nwsamples[name])
    nw = Nw[name]
    #pl.figure()
    #pl.hist(nwsamp,bins=np.linspace(0,10,50),alpha=0.5,histtype='stepfilled')
    #pl.xlabel('$N_w$')
    #pl.title(name)
    #pl.axvline(nw)
    for ii in range(len(nwsamp)):
        allsamples.append(nwsamp[ii])
    allseels.append(nw)

pl.figure()
pl.hist(allsamples,bins=np.linspace(0,10,50),normed=True,alpha=0.5,label='samples')
pl.hist(allseels,bins=np.linspace(0,10,50),normed=True,alpha=0.5,label='SEELs')
pl.legend(loc='upper right')
pl.xlabel('$N_w$')

from scipy.stats import ks_2samp as ks
print ks(allsamples, allseels)
# null hypothesis: same distribution

# now do it for individual seels
for name in sz.keys():
    if name == 'J1248':
        continue
    nwsamp = np.array(Nwsamples[name])
    nw = Nw[name]
    m,s = clip(nwsamp)
    print name, (m-nw)/s
    print '%.2f'%ks(nwsamp,np.array([nw]))[1]

# KS test: the test statistic would be expected with probability equal to the p-value.
# p-value = probability of obtaining a result equal to or more extreme than what was actually observed, assuming that the null hypothesis is true. You have to first choose a significance level (5% or 1%) and then if the p-value is less than or equal to the chosed significance level (alpha), the test suggests that the observed data are inconsistent with the null hypothesis, so that the null hypothesis must be rejected. 
