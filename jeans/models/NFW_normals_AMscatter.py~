from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from jeans.makemodel import *
from astLib import astCalc
from imageSim import SBObjects
from itertools import product
from linslens import EELsModels_huge as L

def run(Mstar,result,name):
    
    # get a new size from vdW14
    logrmu = 0.42 - 0.71*(10.+np.log10(5.)) + 0.71*Mstar
    logr = np.random.normal(loc=logrmu,scale=0.11)

    # make model and extract useful properties
    model = L.EELs(result,name)
    model.Initialise()
    RE1,_ = model.GetSourceSize(kpc=True)
    #print RE1, 10**logr
    RE = 10**logr
    fits = model.fits
    scale = model.scale
    z = model.z

    r = np.logspace(-5,5,1501)
    gal1 = model.srcs[0]
    gal1.re = RE # in KPC
    #gal1.setAmpFromMag(mag,ZPdic[bands[name]]) # but this doesn't matter
    sb = gal1.eval(r)
    
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
    sigma_star = veldisp(r,sb,Mlum+Mdm,ap=1.0*RE)
    #vd = (sigma_dm + sigma_star)**0.5
    vd = sigma_star**0.5
    vlum, vdm = veldisp(r,sb,Mlum,ap=1.0*RE)**0.5, veldisp(r,sb,Mdm,ap=1.0*RE)**0.5

    # SB
    VD.append([RE,vdm,vlum])
    cat_nugg.append([RE,vd])
    print  name, '%.2f'%vd,RE

masses = np.load('/data/ljo31b/EELs/inference/new/huge/masses_211.npy')
logM = masses[3]

phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge_new_new.fits')[1].data
names = phot['name']
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
ZPdic = dict([('F555W',25.711),('F606W',26.493),('F814W',25.947)])

cat_nugg = []
VD = []

dir = '/data/ljo31/Lens/LensModels/twoband/'


for i in range(logM.size):
    name = names[i]
    if name == 'J1606':
        continue
    result = np.load(dir+names[i]+'_211')
    
    M = logM[i]
    run(M,result,name)

np.savetxt('/data/ljo31b/EELs/phys_models/models/NFW_normal_ap1re.dat',cat_nugg)


np.savetxt('/data/ljo31/Lens/pylathon/FP/data4_ap1re.dat',VD)
re,vdd,vdl = np.loadtxt('/data/ljo31/Lens/pylathon/FP/data4_ap1re.dat').T

pl.figure()
pl.scatter(np.log10(vdd/100.),np.log10(re),color='DarkGray',label='dark')
pl.scatter(np.log10(vdl/100.),np.log10(re),color='k',label='light')
pl.scatter(0.5*np.log10(vdd/100.) + 0.5*np.log10(vdl/100.), np.log10(re),color='b',label='total')
pl.ylabel('log re')
pl.xlabel('log sigma')
pl.legend(loc='upper left')
pl.show()
