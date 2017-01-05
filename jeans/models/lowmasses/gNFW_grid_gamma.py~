from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from jeans.makemodel import *
from astLib import astCalc
from imageSim import SBObjects
from itertools import product
from linslens import EELsModels_huge as L
import ndinterp
from multiprocessing import Pool

def run(Mstar,result,name):

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
    Mhalo = highz_halo_table(Mstar,0.5)
    rhalo = virialRadius(Mhalo,z)    

    # stellar mass profile
    lr,light = deproject(r,sb)
    Mlum = light2mass(lr,light,1.)
    fac = Mlum[-1]/10**Mstar
    Mlum /= fac

    arr = [[lr,rhalo,Mhalo,gammagrid[n],z] for n in range(len(gammagrid))]

    Mdm = np.zeros((lr.size,gammagrid.size))
    sigma_star = np.zeros(gammagrid.size)
    out = p.map(gridgNFW2,arr)
    for i in range(len(arr)):
        Mdm[:,idx[i]] = out[i]
    
    # also multiprocess sigma star!
    arr = [[r,sb,Mlum+Mdm[:,idx[i]],1.5*RE] for i in range(len(arr))]
    out = p.map(gridveldisp,arr)
    for i in range(len(arr)):
        sigma_star[idx[i]] = out[i]

    vd = sigma_star**0.5

    # SB
    print name
    return vd

masses = np.load('/data/ljo31b/EELs/inference/new/huge/masses_212.npy')
logM = masses[3]

phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge_new_new.fits')[1].data
names = phot['name']

cat_nugg = []

dir = '/data/ljo31/Lens/LensModels/twoband/'

gammagrid = np.arange(0,2.5,0.1)
idx = list(range(len(gammagrid)))
p = Pool(8)

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
    vd = run(M,result,name)
    
    # build interplator
    ax = {}
    ax[0] = splrep(gammagrid,np.arange(gammagrid.size),k=1,s=0)
    obj = ndinterp.ndInterp(ax,vd,order=3)
    np.save('/data/ljo31b/EELs/phys_models/models/interpolators/gNFW_gamma_12_'+name,obj)



