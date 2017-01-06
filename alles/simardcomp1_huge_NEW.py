import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import cPickle
from tools.EllipsePlot import *
from astLib import astCalc

m,l,h = np.load('/data/ljo31/Lens/LensParams/Structure_2src_huge_new.npy')
m1,l1,h1 = np.load('/data/ljo31/Lens/LensParams/Structure_1src_huge_new.npy')
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
name = m.keys()

# first, focus on 211 models
re,n,dre,dn = [],[],[],[]
q = []
names = []
for name in m.keys():
    Da = astCalc.da(sz[name])
    scale = Da*1e3*np.pi/180./3600.
    n.append(m1[name]['Source 1 n'])
    re.append(m1[name]['Source 1 re']*0.05*scale)
    dn.append(l1[name]['Source 1 n'])
    dre.append(l1[name]['Source 1 re']*0.05*scale)
    q.append(m1[name]['Source 1 q'])
    names.append(name)

n,re,dn,dre =np.array(n), np.array(re), np.array(dn), np.array(dre)
q = np.array(q)

# plot table
for ii in range(len(n)):
    print names[ii], '& $', '%.2f'%n[ii], '\pm', '%.2f'%dn[ii], '$'

# 2-component models
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



print r'\begin{tabular}{c|ccc}\hline'
print r'lens & $n$ (X model) & $n_{disk} (Y model) & $n_{bulge}$ (Y model) \\\hline'
# total table
for ii in range(len(n)):
    print names[ii], '& $', '%.2f'%n[ii], '\pm', '%.2f'%dn[ii], '$ & $', '%.2f'%n1[ii], '\pm', '%.2f'%dn1[ii], '$ & $', '%.2f'%n2[ii], '\pm', '%.2f'%dn2[ii], r'$ \\'
    

### from here on
###
###
###
cat = py.open('/data/ljo31b/EELs/catalogues/asu3.fits')[1].data

PpS = cat['PpS'] # prob B+D is not required
BT = cat['__B_T_g']
eBT = cat['e__B_T_g']
ii = np.where((eBT<0.1)&(BT>0.05)&(PpS>0.05)&(PpS<0.95))

PpS = PpS[ii]
BT = BT[ii]

# also look at 2-component models. What is the distribution of Sersic indices there? And what fraction NEED two-component sources? 
# What about the non-nugget sources? They are only marginally less compact.
BTeels = np.array([0.43,0.11,0.61,0.33,0.57,0.76,0.21,0.41,0.51])
BTeels = np.array([0.72,0.71,0.61,0.47,0.72,0.26,0.44,0.52])
pl.figure()
pl.hist(BT[PpS<0.3],25,normed=1,histtype='stepfilled',label='Simard+11 \n B+D',alpha=0.6)
pl.hist(BT[PpS>0.3],25,normed=1,histtype='stepfilled',label='Simard+11 \n spheroidal',alpha=0.6)

for ii in range(len(BTeels)-1):
    pl.axvline(BTeels[ii],ls='--',lw=3,color='k')
pl.axvline(BTeels[-1],ls='--',lw=3,color='k',label='EELs')
pl.xlim([0,1])
pl.xlabel('$B/T$ (Y models)')
pl.legend(loc='upper left')
pl.savefig('/data/ljo31/Lens/TeXstuff/paper/nhistBTeelsHUGETWO.pdf')
# add our BT ratios!

from scipy.stats import ks_2samp as ks
print 'non-BT',ks(BT[PpS>0.3],BTeels)[1]
print 'needs BT','%.3f'%ks(BT[PpS<0.3],BTeels)[1]
