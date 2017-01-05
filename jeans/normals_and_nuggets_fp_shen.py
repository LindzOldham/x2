from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from makemodel import *
from astLib import astCalc
from imageSim import SBObjects
from itertools import product

def run(Mstar,n):
    # shen 2003's size-mass relation
    logr = np.log10(2.88e-6) + 0.56*Mstar
    re_norm = 10**logr
    re_nugg = 0.5*re_norm
    #re_nugg = 10**(Mstar-10.7)
    print '%.2f'%Mstar, '%.2f'%n, '%.2f'%re_norm, '%.2f'%re_nugg

    # construct nugget and normal galaxy - both have the same stellar mass and sit in identical haloes
    gal_norm = SBObjects.Sersic('normal',{'x':0,'y':0,'re':re_norm,'n':n,'q':1,'pa':0,'amp':1.})
    gal_nugg = SBObjects.Sersic('nugget',{'x':0,'y':0,'re':re_nugg,'n':n,'q':1,'pa':0,'amp':1.})

    # get halo mass at z=0.5
    Mhalo = buildhalo(10**Mstar)
    rhalo = virialRadius(Mhalo,0.5)    

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

    # take sigma within the effective radius of the galaxy
    sigma_dm_norm = veldisp(r,sb_norm,Mdm,ap=0.25*gal_norm.re)
    sigma_dm_nugg = veldisp(r,sb_nugg,Mdm,ap=0.25*gal_nugg.re)
    sigma_star_norm = veldisp(r,sb_norm,Mlum_norm,ap=0.25*gal_norm.re)      
    sigma_star_nugg = veldisp(r,sb_nugg,Mlum_nugg,ap=0.25*gal_nugg.re)  
    vd_norm = (sigma_dm_norm + sigma_star_norm)**0.5
    vd_nugg = (sigma_dm_nugg + sigma_star_nugg)**0.5

    # m/l = 1
    mag_sol = solarmag.getmag('F606W_ACS',0.5)
    mag = -2.5*Mstar + mag_sol
    
    # SB
    scale = astCalc.da(0.5)*1e3 * np.pi/180./3600.
    re_norm_arcsec = re_norm / scale
    re_nugg_arcsec = re_nugg / scale
    mu_norm = mag + 2.5*np.log10(2.*np.pi*re_norm_arcsec**2.)
    mu_nugg = mag + 2.5*np.log10(2.*np.pi*re_nugg_arcsec**2.)
    cat_norm.append([re_norm, vd_norm, mu_norm])
    cat_nugg.append([re_nugg, vd_nugg, mu_nugg])
    print '%.2f'%mag, '%.2f'%mu_norm, '%.2f'%mu_nugg

cat_norm, cat_nugg = [], []
sersic_indices = np.random.uniform(low=1.,high=6.,size=100)
Mstars = np.random.uniform(low=10.5,high=11.5,size=100)

# now if these galaxies all have stellar-mass-to-light ratios of unity, we can give them magnitudes and SBs and hence populate their fundamental planes! To populate the planes, we should generate a random sample drawn from uniform $n$, $M_{\star}$ distributions I think.



for i in range(Mstars.size):
    run(Mstars[i],n=sersic_indices[i])

np.savetxt('/data/ljo31b/EELs/phys_models/FP_normals_100_quarterre_shen.dat',cat_norm)
np.savetxt('/data/ljo31b/EELs/phys_models/FP_nuggets_100_quarterre_shen.dat',cat_nugg)


