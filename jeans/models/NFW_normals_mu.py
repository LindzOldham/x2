from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from jeans.makemodel import *
from astLib import astCalc
from imageSim import SBObjects
from itertools import product
from linslens import EELsModels_huge as L

def run(Mstar_old,result,name):
    
    # make model and extract useful properties
    model = L.EELs(result,name)
    model.Initialise()
    re,_ = model.GetSourceSize(kpc=True)
    scale = model.scale
    z = model.z
    fits = model.fits
    mag = model.GetIntrinsicMags()[0]
    mu = model.GetSB()[0]

    # (1) keep mu, re, recalculate Mstar
    Mstarmu = np.log10(re) - np.log10(2.88e-6)
    Mstarmu /= 0.56
    Mstar = np.random.normal(loc=Mstarmu,scale=0.3)
    print Mstar, Mstarmu
    #print 'Mtar old/new', Mstar_old, Mstar
    #Mstar = Mstar_old

    # (2) keep mu, Mstar, recalculate re
    #logr = np.log10(2.88e-6) + 0.56*Mstar
    #re = 10**logr
    
    r = np.logspace(-5,5,1501)
    if name in ['J0901','J1218','J1323']:
        gal1 = model.srcs[0]
        gal1.re *= 0.05*scale
        sb = fits[0][-2]*gal1.eval(r)

    elif name == 'J0837':
        gal1 = model.srcs[0]
        gal1.re *= 0.05*scale
        sb = fits[0][-3]*gal1.eval(r) # the other one is the dust lane!

    else:
        gal1,gal2 = model.srcs
        gal1.re *= 0.05*scale
        gal2.re *= 0.05*scale
        sb = fits[0][-3]*gal1.eval(r) + fits[0][-2]*gal2.eval(r)# in image units, but is normalised by the total mass

    # halo properties
    Mhalo = highz_halo_table(Mstar,0.5)
    rhalo = virialRadius(Mhalo,z)    

    # stellar mass profile
    lr,light = deproject(r,sb)
    Mdm = NFW(lr,rhalo,Mhalo,z=z)
    Mlum = light2mass(lr,light,1.)
    fac = Mlum[-1]/10**Mstar
    Mlum /= fac
    
    # take sigma within the effective radius of the galaxy
    vlum, vdm = veldisp(r,sb,Mlum,ap=1.5*re)**0.5, veldisp(r,sb,Mdm,ap=1.5*re)**0.5
    print name, '%.2f'%vlum, '%.2f'%vdm,'%.2f'%(vlum/vdm)
    sigma_star = veldisp(r,sb,Mlum+Mdm,ap=1*re)
    #vd = (sigma_dm + sigma_star)**0.5
    vd = sigma_star**0.5

    # SB
    VD.append([re,vlum,vdm])
    cat_nugg.append(vd)
    #print  name, '%.2f'%vd, mu,re

masses = np.load('/data/ljo31b/EELs/inference/new/huge/masses_212.npy')
logM = masses[3]

phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge_new_new.fits')[1].data
names = phot['name']
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
ZPdic = dict([('F555W',25.711),('F606W',26.493),('F814W',25.947)])

cat_nugg = []

dir = '/data/ljo31/Lens/LensModels/twoband/'
VD = []

for i in range(logM.size):
    name = names[i]
    if name in ['J0913','J1125','J1144','J1347','J1446','J1605','J1619','J2228']:
        result = np.load(dir+names[i]+'_212')
    elif name in ['J0837','J0901','J1218','J1323']:
        result = np.load(dir+names[i]+'_211')
    elif name == 'J1606':
        continue
    else:
        print 'missing eel!'
    
    M = logM[i]
    run(M,result,name)

np.savetxt('/data/ljo31b/EELs/phys_models/models/NFW_normal_newMSTAR.dat',cat_nugg)

np.savetxt('/data/ljo31/Lens/pylathon/FP/data3.dat',VD)
re,vdl,vdd = np.loadtxt('/data/ljo31/Lens/pylathon/FP/data3.dat').T

pl.figure()
pl.scatter(np.log10(vdd/100.),np.log10(re),color='DarkGray',label='dark')
pl.scatter(np.log10(vdl/100.),np.log10(re),color='k',label='light')
pl.scatter(0.5*np.log10(vdd/100.) + 0.5*np.log10(vdl/100.), np.log10(re),color='b',label='total')
pl.ylabel('log re')
pl.xlabel('log sigma')
pl.legend(loc='upper left')
pl.show()
