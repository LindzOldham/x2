import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import cPickle
from tools.EllipsePlot import *
from astLib import astCalc

sz = dict([('J0837',0.6411),('J0901',0.586),('J0913',0.539),('J1125',0.689),('J1144',0.706),('J1218',0.6009),('J1248',0.528),('J1323',0.4641),('J1347',0.63),('J1446',0.585),('J1605',0.542),('J1606',0.6549),('J1619',0.6137),('J2228',0.4370)])


''' select as RNs any objects that are more compact than van Dokkum's criterion (the most stringent) for both 1- and 2-component models. '''
logR,logM,dlogR,dlogM,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_2src.npy').T
logR1,logM1,dlogR1,dlogM1,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_1src.npy').T
name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']

def vD(x,y):
    return x-10.7

def select(logM,logR,logM1,logR1):
    line = vD(logM,logR)
    line1 = vD(logM1,logR1)
    nuggets = np.where((logR<line) & (logR1<line1))
    return nuggets

nuggets = select(logM,logR,logM1,logR1)
logR,logM,dlogR,dlogM,rho = logR[nuggets],logM[nuggets],dlogR[nuggets],dlogM[nuggets],rho[nuggets]
logR1,logM1,dlogR1,dlogM1 = logR1[nuggets],logM1[nuggets],dlogR1[nuggets],dlogM1[nuggets]
name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name'][nuggets]

m,l,h = np.load('/data/ljo31/Lens/LensParams/Structure_2src.npy')
m1,l1,h1 = np.load('/data/ljo31/Lens/LensParams/Structure_1src.npy')

n1,n2,re1,re2 = logR*0.,logR*0.,logR*0.,logR*0.
dn1,dn2,dre1,dre2 = logR*0.,logR*0.,logR*0.,logR*0.
n,re,dn,dre = logR*0.,logR*0.,logR*0.,logR*0.
for ii in range(name.size):
    Da = astCalc.da(sz[name[ii]])
    scale = Da*1e3*np.pi/180./3600.
    if name[ii] in m.keys():
        n1[ii],n2[ii],re1[ii],re2[ii] = m[name[ii]]['Source 1 n'], m[name[ii]]['Source 2 n'], scale*0.05*m[name[ii]]['Source 1 re'], scale*0.05*m[name[ii]]['Source 2 re'] 
        dn1[ii],dn2[ii],dre1[ii],dre2[ii] = l[name[ii]]['Source 1 n'], l[name[ii]]['Source 2 n'], scale*0.05*l[name[ii]]['Source 1 re'], scale*0.05*l[name[ii]]['Source 2 re']        
    n[ii],dn[ii],re[ii],dre[ii] = m1[name[ii]]['Source 1 n'], l1[name[ii]]['Source 1 n'], scale*0.05*m1[name[ii]]['Source 1 re'], scale*0.05*l1[name[ii]]['Source 1 re'] 

# now make a table with everything on it
for ii in range(name.size):
    if name[ii] in m.keys():
        print name[ii], '& $', '%.2f'%logR[ii], ' \pm ', '%.2f'%dlogR[ii], '$ & $', '%.2f'%logM[ii], ' \pm ', '%.2f'%dlogM[ii], '$ & $', '%.2f'%n1[ii], ' \pm ', '%.2f'%dn1[ii], '$ & $', '%.2f'%re1[ii], ' \pm ', '%.2f'%dre1[ii], '$ & $','%.2f'%n2[ii], ' \pm ', '%.2f'%dn2[ii], '$ & $', '%.2f'%re2[ii], ' \pm ', '%.2f'%dre2[ii], '$ & $', '%.2f'%n[ii], ' \pm ', '%.2f'%dn[ii], '$ & $', '%.2f'%re[ii], ' \pm ', '%.2f'%dre[ii], r'$ \\'
    else:
        print name[ii], '& $', '%.2f'%logR[ii], ' \pm ', '%.2f'%dlogR[ii], '$ & $', '%.2f'%logM[ii], ' \pm ', '%.2f'%dlogM[ii], '$ & -- & -- & -- & -- & $', '%.2f'%n[ii], ' \pm ', '%.2f'%dn[ii], '$ & $', '%.2f'%re[ii], ' \pm ', '%.2f'%dre[ii], r'$ \\'

np.save('/data/ljo31/Lens/LensParams/structurecat_srcs_2src',np.column_stack((n1,n2)))
np.save('/data/ljo31/Lens/LensParams/structurecat_srcs_1src',n)
