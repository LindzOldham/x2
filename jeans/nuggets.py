from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from makemodel import *
from astLib import astCalc
from imageSim import SBObjects
from itertools import product

def run(Mstar=10.5,n=4.):
    # van dokkum 2015's compactness criterion
    re = 10**(Mstar-10.7)
    print '%.2f'%Mstar, '%.2f'%n, '%.2f'%re

    # construct nugget
    kpc2cm = 3.086e21
    gal = SBObjects.Sersic('nugget',{'x':0,'y':0,'re':re,'n':n,'q':1,'pa':0,'amp':1.})

    # get halo mass at z=0.5
    Mhalo = buildhalo(10**Mstar)
    rhalo = virialRadius(Mhalo,0.5)    

    # mass components
    r = np.logspace(-5,5,1501)
    sb = gal.eval(r)
    lr,light = deproject(r,sb)
    Mdm = NFW(lr,rhalo,Mhalo)
    Mlum = light2mass(lr,light,1.)
    fac = Mlum[-1]/10**Mstar
    Mlum /= fac

    # define apertures as multiples of the effective radius
    aps = np.logspace(-1.5,0.7,50)
    sigma_dm = veldisp(r,sb,Mdm,ap=aps*gal.re)
    sigma_star = veldisp(r,sb,Mlum,ap=aps*gal.re)        
    vd = (sigma_dm + sigma_star)**0.5

    pl.figure()
    pl.xlabel('aperture radius / mutiples of $r_e$')
    pl.ylabel('$\sigma$ / kms$^{-1}$')
    pl.title('Mstar = '+'%.2f'%Mstar)
    pl.plot(aps, vd)
    pl.xlim([0,5])
    #pl.ylim([160,230])
    pl.savefig('/data/ljo31/public_html/Lens/phys_models/synthetic_nuggets/n=6/Mstar_'+'%.2f'%Mstar+'_n_'+'%.2f'%n+'.pdf')
    cat.append([Mstar,n,re,Mhalo,vd])

cat = []
sersic_indices = np.linspace(1.,6.,20)
Mstars = np.linspace(10.5,12,20)

for i in range(Mstars.size):
    run(Mstars[i],n=6.)

f=open('/data/ljo31/public_html/Lens/phys_models/synthetic_nuggets/n=6/cat','wb')
cPickle.dump(cat,f,2)
f.close()

'''for i in range(Mstars.size):
    for j in range(sersic_indices.size):
        run(Mstars[i],sersic_indices[j])'''

