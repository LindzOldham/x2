import numpy as np, pylab as pl

result = np.load('/data/ljo31/Lens/J2228/twoband_superbig')
lp,trace,dic,_= result
###
presult = np.load('/data/ljo31/Lens/LensModels/twoband/J2228_211')
plp,ptrace,pdic,_= presult

for key in ['Source 1 n','Source 1 re','Galaxy 1 n','Galaxy 1 re','Galaxy 2 n','Galaxy 2 re']:
    pl.figure()
    pl.subplot(211)
    pl.hist(dic[key][:,0].ravel(),30,alpha=0.5,histtype='stepfilled',normed=True)
    pl.hist(pdic[key][:,0].ravel(),30,alpha=0.5,histtype='stepfilled',normed=True)
    pl.title(key)
    pl.subplot(212)
    pl.plot(dic[key][:,0])
