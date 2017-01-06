import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import cPickle
from tools.EllipsePlot import *
from astLib import astCalc

sz = dict([('J0837',0.4256),('J0901',0.311),('J0913',0.395),('J1125',0.442),('J1144',0.372),('J1218',0.3182),('J1248',0.304),('J1323',0.3194),('J1347',0.39),('J1446',0.317),('J1605',0.306),('J1606',0.3816),('J1619',0.3638),('J2228',0.2391)])

''' select as RNs any objects that are more compact than van Dokkum's criterion (the most stringent) for both 1- and 2-component models. '''
logR,logM,dlogR,dlogM,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_lensgals_2src.npy').T
logR1,logM1,dlogR1,dlogM1,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_lensgals_1src.npy').T
name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']

m,l,h = np.load('/data/ljo31/Lens/LensParams/Structure_lensgals_1src.npy')

n1,n2,re1,re2 = logR*0.,logR*0.,logR*0.,logR*0.
dn1,dn2,dre1,dre2 = logR*0.,logR*0.,logR*0.,logR*0.
for ii in range(name.size):
    Da = astCalc.da(sz[name[ii]])
    scale = Da*1e3*np.pi/180./3600.
    if 'Galaxy 2 n' in m[name[ii]].keys():
        n1[ii],n2[ii],re1[ii],re2[ii] = m[name[ii]]['Galaxy 1 n'], m[name[ii]]['Galaxy 2 n'], scale*0.05*m[name[ii]]['Galaxy 1 re'], scale*0.05*m[name[ii]]['Galaxy 2 re'] 
        dn1[ii],dn2[ii],dre1[ii],dre2[ii] = l[name[ii]]['Galaxy 1 n'], l[name[ii]]['Galaxy 2 n'], scale*0.05*l[name[ii]]['Galaxy 1 re'], scale*0.05*l[name[ii]]['Galaxy 2 re']  
    else:
        n1[ii],re1[ii] = m[name[ii]]['Galaxy 1 n'], scale*0.05*m[name[ii]]['Galaxy 1 re']
        dn1[ii],dre1[ii] = l[name[ii]]['Galaxy 1 n'],  scale*0.05*l[name[ii]]['Galaxy 1 re']  
   
# now make a table with everything on it
for ii in range(name.size):
    if 'Galaxy 2 n' in m[name[ii]].keys():
        print name[ii], '& $', '%.2f'%logR[ii], ' \pm ', '%.2f'%dlogR[ii], '$ & $', '%.2f'%logM[ii], ' \pm ', '%.2f'%dlogM[ii], '$ & $', '%.2f'%n1[ii], ' \pm ', '%.2f'%dn1[ii], '$ & $', '%.2f'%re1[ii], ' \pm ', '%.2f'%dre1[ii], '$ & $','%.2f'%n2[ii], ' \pm ', '%.2f'%dn2[ii], '$ & $', '%.2f'%re2[ii], ' \pm ', '%.2f'%dre2[ii], r'$ \\'
    else:
        print name[ii], '& $', '%.2f'%logR[ii], ' \pm ', '%.2f'%dlogR[ii], '$ & $', '%.2f'%logM[ii], ' \pm ', '%.2f'%dlogM[ii], '$ & $', '%.2f'%n1[ii], ' \pm ', '%.2f'%dn1[ii], '$ & $', '%.2f'%re1[ii], ' \pm ', '%.2f'%dre1[ii], '$ & -- & -- ', r'$ \\'

np.save('/data/ljo31/Lens/LensParams/structurecat_lensgals',np.column_stack((n1,n2)))
