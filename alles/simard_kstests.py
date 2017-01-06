import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import cPickle
from tools.EllipsePlot import *
from astLib import astCalc

### test 1: compare our X models with Simard's X models. Likely to be drawn from same disrtribution?

### load in seels data
m,l,h = np.load('/data/ljo31/Lens/LensParams/Structure_2src_huge_new.npy')
m1,l1,h1 = np.load('/data/ljo31/Lens/LensParams/Structure_1src_huge_new.npy')
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
NAME = m.keys()
NAME.sort()
ns = np.array([m1[name]['Source 1 n'] for name in NAME])
qs = np.array([m1[name]['Source 1 q'] for name in NAME])

### load in simard data
cat = py.open('/data/ljo31b/EELs/catalogues/joinedsimtabs.fits')[1].data
ng = cat['ng3'] # pure sersic fit

### compare ng with n
from scipy.stats import ks_2samp as ks
for name in NAME:
    n = m1[name]['Source 1 n']
    print name, n
    print '%.2f'%ks(ng,np.array([n]))[1]

print ks(ng,ns)[1] # not significant




PpS = cat['PpS1'] # prob B+D is not required
BT = cat['__B_T_g1']
BTeels = np.array([0.72,0.71,0.61,0.47,0.72,0.26,0.44,0.52])

# ks test: drawn from pop needing BT?
print 'non-BT',ks(BT[PpS>0.3],BTeels)[1]
print 'needs BT','%.3f'%ks(BT[PpS<0.3],BTeels)[1]



# finally, axis ratios / ellipticities
e = py.open('/data/ljo31b/EELs/catalogues/ellipt.fits')[1].data['e']
e = e[np.isnan(e)==False]
eeels = 1-qs

print ks(e,eeels)[1]

## also n-distribution of 2-component models
re1,re2,n1,n2,dre1,dre2,dn1,dn2 = [],[],[],[],[],[],[],[]
for name in m.keys():
    print name
    conditions = ['J0837' in name, 'J0901' in name, 'J1218' in name]
    Da = astCalc.da(sz[name])
    scale = Da*1e3*np.pi/180./3600.
    if any(conditions):
        print 'bad'
        n1.append(0)
        re1.append(0)
        dn1.append(0)
        dre1.append(0)
        n2.append(0)
        re2.append(0)
        dn2.append(0)
        dre2.append(0)
        continue
    n1.append(m[name]['Source 1 n'])
    re1.append(m[name]['Source 1 re']*0.05*scale)
    dn1.append(l[name]['Source 1 n'])
    dre1.append(l[name]['Source 1 re']*0.05*scale)
    ###
    n2.append(m[name]['Source 2 n'])
    re2.append(m[name]['Source 2 re']*0.05*scale)
    dn2.append(l[name]['Source 2 n'])
    dre2.append(l[name]['Source 2 re']*0.05*scale)

# order: disk is n1, bulge is n2
for ii in range(len(n1)):
    a,b = n1[ii],n2[ii]
    n1[ii], n2[ii] = np.sort((a,b))
    print '%.2f'%a,'%.2f'%b,'%.2f'%n1[ii],'%.2f'%n2[ii]

nb2 = cat['nb2']
n2 = np.array(n2)
print ks(nb2,n2[n2>0])
