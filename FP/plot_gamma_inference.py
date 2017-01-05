import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances
from SampleOpt import AMAOpt
from astLib import astCalc
from scipy.misc import logsumexp


outFile = '/data/ljo31b/EELs/FP/inference/FP_infer_gNFW_gamma_HYPER_LO_M_salpeter'
result = np.load(outFile)
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)

pl.figure()
pl.plot(lp[2000:])

bins = np.arange(0.3,1.5,0.05)

pl.figure()
pl.hist(dic['gamma'][1000:].ravel(),bins,histtype='stepfilled',color='Crimson',alpha=0.5,label='Salpeter',normed=True,edgecolor='none')
pl.xlabel('gNFW inner slope')
pl.axvline(1.,color='k')

g = dic['gamma'][1000:].ravel()
gamma, gl, gu = np.median(g), np.percentile(g,16),np.percentile(g,84)
print '%.2f'%gamma, '%.2f'%(gamma-gl), '%.2f'%(gu-gamma)

g = dic['tau'][1000:].ravel()
gamma, gl, gu = np.median(g), np.percentile(g,16),np.percentile(g,84)
print '%.2f'%gamma, '%.2f'%(gamma-gl), '%.2f'%(gu-gamma)


outFile = '/data/ljo31b/EELs/FP/inference/FP_infer_gNFW_gamma_HYPER_LO_M'
result = np.load(outFile)
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)

pl.hist(dic['gamma'][1000:].ravel(),bins,histtype='stepfilled',color='SteelBlue',alpha=0.5,label='Chabrier',normed=True,edgecolor='none')
pl.legend(loc='upper left')
pl.figtext(0.535,0.8,'NFW',rotation=90)

pl.show()

g = dic['gamma'][1000:].ravel()
gamma, gl, gu = np.median(g), np.percentile(g,16),np.percentile(g,84)
print '%.2f'%gamma, '%.2f'%(gamma-gl), '%.2f'%(gu-gamma)

g = dic['tau'][1000:].ravel()
gamma, gl, gu = np.median(g), np.percentile(g,16),np.percentile(g,84)
print '%.2f'%gamma, '%.2f'%(gamma-gl), '%.2f'%(gu-gamma)
