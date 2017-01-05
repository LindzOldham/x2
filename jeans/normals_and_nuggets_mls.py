from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from makemodel import *
from astLib import astCalc
from imageSim import SBObjects
from itertools import product

''' the aim this time is to look at how mstar, mdyn and mtot scale with the FP parameters. I think it's because the size-mass plane is steeper that m/l scales like this or summin'''

def run(Mstar=10.5,n=4.):
    # shen 2003's size-mass relation
    logr = np.log10(2.88e-6) + 0.56*Mstar
    re_norm = 10**logr
    ## draw nuggets from the EELs mass-size relation
    logr = 1.05*Mstar - 11.36
    re_nugg = 10**logr
    print '%.2f'%Mstar, '%.2f'%n, '%.2f'%re_norm, '%.2f'%re_nugg

    # construct nugget and normal galaxy - both have the same stellar mass and sit in identical haloes
    gal_norm = SBObjects.Sersic('normal',{'x':0,'y':0,'re':re_norm,'n':n,'q':1,'pa':0,'amp':1.})
    gal_nugg = SBObjects.Sersic('nugget',{'x':0,'y':0,'re':re_nugg,'n':n,'q':1,'pa':0,'amp':1.})

    # get halo mass at z=0.5
    Mhalo = buildhalo(10**Mstar)
    rhalo = virialRadius(Mhalo,0.5)    
    # get halo mass within the effective radius
    Mhalo_nugg = NFW(re_nugg, rhalo, Mhalo)
    Mhalo_norm = NFW(re_norm, rhalo, Mhalo)


    # mass components - normal galaxy
    r = np.logspace(-5,5,1501)
    sb_norm = gal_norm.eval(r)
    lr,light_norm = deproject(r,sb_norm)
    Mdm = NFW(lr,rhalo,Mhalo)
    Mlum_norm = light2mass(lr,light_norm,1.)
    fac = Mlum_norm[-1]/10**Mstar
    Mlum_norm /= fac

    # mass component - nugget
    sb_nugg = gal_nugg.eval(r)
    lr,light_nugg = deproject(r,sb_nugg)
    Mlum_nugg = light2mass(lr,light_nugg,1.)
    fac = Mlum_nugg[-1]/10**Mstar
    Mlum_nugg /= fac

    # define apertures in kiloparsec, so that both galaxies are being sampled in the same apertures
    sigma_dm_norm = veldisp(r,sb_norm,Mdm,ap=gal_norm.re)
    sigma_dm_nugg = veldisp(r,sb_nugg,Mdm,ap=gal_nugg.re)
    sigma_star_norm = veldisp(r,sb_norm,Mlum_norm,ap=gal_norm.re)      
    sigma_star_nugg = veldisp(r,sb_nugg,Mlum_nugg,ap=gal_nugg.re)  
    vd_norm = (sigma_dm_norm + sigma_star_norm)**0.5
    vd_nugg = (sigma_dm_nugg + sigma_star_nugg)**0.5
    print vd_nugg
    # save: Mstar, Mhalo, rhalo, re, sigma

    
    cat.append([Mstar,n,re_norm, re_nugg,rhalo,Mhalo,vd_norm, vd_nugg, Mhalo_nugg, Mhalo_norm,sigma_dm_nugg,sigma_star_nugg,sigma_dm_norm,sigma_star_norm])

cat = []
sersic_indices = np.linspace(1.,6.,20)
Mstars = np.linspace(10.5,11.6,20)

for i in range(Mstars.size):
    run(Mstars[i],n=4.)

f=open('/data/ljo31/public_html/Lens/phys_models/synthetic_both/n=6/ML_CATALOGUE','wb')
cPickle.dump(cat,f,2)
f.close()
'''for i in range(Mstars.size):
    for j in range(sersic_indices.size):
        run(Mstars[i],sersic_indices[j])'''

