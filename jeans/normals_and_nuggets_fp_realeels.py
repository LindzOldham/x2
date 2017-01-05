from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from makemodel import *
from astLib import astCalc
from imageSim import SBObjects
from itertools import product

def run(Mstar,n,re_nugg,q,scale,z,mu_nugg,band):

    # construct nugget and normal galaxy - both have the same stellar mass and sit in identical haloes
    gal_nugg = SBObjects.Sersic('nugget',{'x':0,'y':0,'re':re_nugg,'n':n,'q':q,'pa':0,'amp':1.})

    # mass to light ratio
    DM = 5. - 5.*np.log10(astCalc.dl(z)*1e6)
    M = mu_nugg - 2.5*np.log10(2.*np.pi*re_nugg**2./scale**2.) + DM
    Msun = solarmag.getmag(band+'_ACS',z)
    L = 0.4*(Msun-M)
    ML = Mstar-L
    ML = 10**ML
    # get halo mass at z=0.5
    Mhalo = buildhalo(10**Mstar)
    print Mhalo
    #if Mhalo>15:
    #    print Mhalo
    #    Mhalo = 14.9
    #Mhalo = highz_halo_table(Mstar,0.5)
    #print Mhalo
    rhalo = virialRadius(Mhalo,z)    

    # mass component - nugget
    r = np.logspace(-5,5,1501)
    sb_nugg = gal_nugg.eval(r)
    lr,light_nugg = deproject(r,sb_nugg)
    Mdm = NFW(lr,rhalo,Mhalo)
    Mlum_nugg = light2mass(lr,light_nugg,1.)
    fac = Mlum_nugg[-1]/10**Mstar
    Mlum_nugg /= fac

    # take sigma within the effective radius of the galaxy
    sigma_dm_nugg = veldisp_cbeta(r,sb_nugg,Mdm,beta=0.,ap=2*gal_nugg.re)
    sigma_star_nugg = veldisp_cbeta(r,sb_nugg,Mlum_nugg,beta=0.,ap=2*gal_nugg.re)
    #sigma_dm_nugg = veldisp(r,sb_nugg,Mdm,ap=3*gal_nugg.re)
    #sigma_star_nugg = veldisp(r,sb_nugg,Mlum_nugg,ap=3*gal_nugg.re)
    #vd_nugg = (sigma_dm_nugg + sigma_star_nugg)**0.5
    vd_nugg = sigma_star_nugg**0.5

    # m/l = 1 - could improve this by working out their MLs
    #mag_sol = solarmag.getmag('F606W_ACS',z)
    #mag = -2.5*Mstar + mag_sol - 5. + 5.*np.log10(astCalc.dl(0.55)*1e6)
    
    # SB
    #re_nugg_arcsec = re_nugg / scale
    #mu_nugg = mag + 2.5*np.log10(2.*np.pi*re_nugg_arcsec**2.)
    cat_nugg.append([re_nugg, vd_nugg, mu_nugg])
    print  '%.2f'%mu_nugg, '%.2f'%re_nugg, '%.2f'%vd_nugg, '%.2f'%n

masses = np.load('/data/ljo31b/EELs/inference/new/huge/masses_211.npy')
logM = masses[3]

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_1.00_lens_vdfit.npy')[()]
phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge_new_new.fits')[1].data
names = phot['name']
table = np.load('/data/ljo31/Lens/LensParams/Structure_1src_huge_new.npy')
re = np.array([table[0][name]['Source 1 re'] for name in names])
ns = np.array([table[0][name]['Source 1 n'] for name in names])
scales = np.array([astCalc.da(sz[name][0])*1e3*np.pi/180./3600. for name in names])
re *= 0.05*scales
qs = np.array([table[0][name]['Source 1 q'] for name in names])
mus = phot['mu v']
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]

cat_norm, cat_nugg = [], []


for i in range(logM.size):
    run(logM[i],ns[i],re[i],qs[i],scales[i],sz[names[i]][0],mus[i],bands[names[i]])

np.savetxt('/data/ljo31b/EELs/phys_models/FP_nuggets_realeels_noDM.dat',cat_nugg)


