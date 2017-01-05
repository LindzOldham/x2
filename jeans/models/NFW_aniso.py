from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from jeans.makemodel import *
from astLib import astCalc
from imageSim import SBObjects
from itertools import product
from linslens import EELsModels_huge as L

def run(Mstar,result,name):

    # e.g.
    #Mstar -= 0.1
    # make model and extract useful properties
    model = L.EELs(result,name)
    model.Initialise()
    RE,_ = model.GetSourceSize(kpc=True)
    fits = model.fits
    scale = model.scale
    z = model.z
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
    #Mhalo = buildhalo(10**Mstar)
    Mhalo = highz_halo_table(Mstar,0.5)
    rhalo = virialRadius(Mhalo,z)    

    # stellar mass profile
    lr,light = deproject(r,sb)
    Mdm = NFW(lr,rhalo,Mhalo,z=z)
    Mlum = light2mass(lr,light,1.)
    fac = Mlum[-1]/10**Mstar
    Mlum /= fac
    
    # take sigma within the effective radius of the galaxy
    #sigma_dm = veldisp(r,sb,Mdm,ap=RE)
    sigma_star = veldisp_cbeta(r,sb,Mlum+Mdm,ap=1.5*RE,beta=0.501)
    #vd = (sigma_dm + sigma_star)**0.5
    vd = sigma_star**0.5

    # SB
    cat_nugg.append(vd)
    print  name, '%.2f'%vd, Mhalo

masses = np.load('/data/ljo31b/EELs/inference/new/huge/masses_212.npy')
logM = masses[3]

phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge_new_new.fits')[1].data
names = phot['name']

cat_nugg = []

dir = '/data/ljo31/Lens/LensModels/twoband/'


for i in range(logM.size-3):
    name = names[i]
    if name in ['J0913','J1125','J1144','J1347','J1446','J1605']:
        result = np.load(dir+names[i]+'_212')
    elif name in ['J0837','J0901','J1218','J1323']:
        result = np.load(dir+names[i]+'_211')
    else:
        print 'missing eel!'

    M = logM[i]
    run(M,result,name)

np.savetxt('/data/ljo31b/EELs/phys_models/models/NFW_radaniso.dat',cat_nugg)

# if no label, it has 1.5 re apertures and beta = -0.5,0 or 0.5 and the true stellar mass!
