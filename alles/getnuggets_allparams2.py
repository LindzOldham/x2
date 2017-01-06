import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import cPickle
from tools.EllipsePlot import *
from astLib import astCalc

lz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]

model_vi,masses,model_vk,ages = np.load('/data/ljo31/Lens/Analysis/InferredAges_1src_all_new.npy')
filename = '/data/ljo31b/EELs/inference/masses.npy'
fi = open(filename)
for i in range(12):
    masses[:,i] = np.load(fi)[3:]

vi,vk,dvi,dvk,v,i,k,Re,dRe = np.load('/data/ljo31/Lens/LensParams/colours_1src.npy')
name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']
logRe = np.log10(Re)
logM = masses[0]
dlogM = np.mean((masses[2],masses[1]),axis=0)
dlogRe = dRe/Re


def vD(x,y):
    return x-10.7

def select(logM,logR):
    line = vD(logM,logR)
    nuggets = np.where(logR<line)
    return nuggets


nuggets = select(logM,logRe)
print nuggets
np.save('/data/ljo31/Lens/LensParams/sourcenuggets',nuggets)

m,l,h = np.load('/data/ljo31/Lens/LensParams/Structure_2src.npy')
m1,l1,h1 = np.load('/data/ljo31/Lens/LensParams/Structure_1src.npy')


SZ = [('Source '+name[ii],sz[name[ii]]) for ii in nuggets[0]]
M = [('Source '+name[ii],m[name[ii]]) for ii in nuggets[0]]
M1 = [('Source '+name[ii],m1[name[ii]]) for ii in nuggets[0]]
L = [('Source '+name[ii],l[name[ii]]) for ii in nuggets[0]]
L1 = [('Source '+name[ii],l1[name[ii]]) for ii in nuggets[0]]
H = [('Source '+name[ii],h[name[ii]]) for ii in nuggets[0]]
H1 = [('Source '+name[ii],h1[name[ii]]) for ii in nuggets[0]]
#M,M1,L,L1,H,H1 = dict(M), dict(M1), dict(L), dict(L1), dict(H), dict(H1)

vi,vk,dvi,dvk,v,i,k = vi[nuggets],vk[nuggets],dvi[nuggets],dvk[nuggets],v[nuggets],i[nuggets],k[nuggets]


np.save('/data/ljo31/Lens/LensParams/sourcenuggets_props',[M,L,H,M1,L1,H1,vi,vk,dvi,dvk,v,i,k,SZ])

# now do the same for lens nuggets
model_vi,masses,model_vk,ages = np.load('/data/ljo31/Lens/Analysis/LensgalInferredAges_1src_all_new.npy')
filename = '/data/ljo31b/EELs/inference/masses.npy'
fi = open(filename)
for i in range(12):
    masses[:,i] = np.load(fi)[:3]

vi,vk,dvi,dvk,v,i,k,Re,dRe = np.load('/data/ljo31/Lens/LensParams/colours_lens_1src.npy')
name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']
logRe = np.log10(Re)
logM = masses[0]
dlogM = np.mean((masses[2],masses[1]),axis=0)
dlogRe = dRe/Re


def vD(x,y):
    return x-10.7

def select(logM,logR):
    line = vD(logM,logR)
    nuggets = np.where(logR<line)
    return nuggets


nuggets = select(logM,logRe)
print nuggets
np.save('/data/ljo31/Lens/LensParams/lensnuggets',nuggets)

m,l,h = np.load('/data/ljo31/Lens/LensParams/Structure_lensgals_2src.npy')
m1,l1,h1 = np.load('/data/ljo31/Lens/LensParams/Structure_lensgals_1src.npy')

LZ = [('Lens '+name[ii],lz[name[ii]]) for ii in nuggets[0]]
M = [('Lens '+name[ii],m[name[ii]]) for ii in nuggets[0]]
M1 = [('Lens '+name[ii],m1[name[ii]]) for ii in nuggets[0]]
L = [('Lens '+name[ii],l[name[ii]]) for ii in nuggets[0]]
L1 = [('Lens '+name[ii],l1[name[ii]]) for ii in nuggets[0]]
H = [('Lens '+name[ii],h[name[ii]]) for ii in nuggets[0]]
H1 = [('Lens '+name[ii],h1[name[ii]]) for ii in nuggets[0]]
#M,M1,L,L1,H,H1 = dict(M), dict(M1), dict(L), dict(L1), dict(H), dict(H1)


vi,vk,dvi,dvk,v,i,k = vi[nuggets],vk[nuggets],dvi[nuggets],dvk[nuggets],v[nuggets],i[nuggets],k[nuggets]


np.save('/data/ljo31/Lens/LensParams/lensnuggets_props',[M,L,H,M1,L1,H1,vi,vk,dvi,dvk,v,i,k,LZ])

m,l,h,m1,l1,h1,vi,vk,dvi,dvk,v,i,k,SZ = np.load('/data/ljo31/Lens/LensParams/sourcenuggets_props.npy')
m3,l3,h3,m13,l13,h13,vi3,vk3,dvi3,dvk3,v3,i3,k3,LZ = np.load('/data/ljo31/Lens/LensParams/lensnuggets_props.npy')

cols = [m,l,h,m1,l1,h1,vi,vk,dvi,dvk,v,i,k,SZ]
col1s = [m3,l3,h3,m13,l13,h13,vi3,vk3,dvi3,dvk3,v3,i3,k3,LZ]

for ii in range(len(cols)):
    cols[ii] = np.concatenate((cols[ii],col1s[ii]))

m,l,h,m1,l1,h1,vi,vk,dvi,dvk,v,i,k,z = cols
m,l,h,m1,l1,h1 = dict(m), dict(l), dict(h), dict(m1), dict(l1), dict(h1)
z = dict(z)

np.save('/data/ljo31/Lens/LensParams/allnuggets_props',cols)


m,l,h,m1,l1,h1,vi,vk,dvi,dvk,v,i,k,z = np.load('/data/ljo31/Lens/LensParams/allnuggets_props.npy')
'''


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
'''
